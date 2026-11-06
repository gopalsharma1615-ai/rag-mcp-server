"""rag-mcp-server: document and knowledge-graph retrieval as MCP tools.

An AI client (Claude Desktop, Claude Code, an IDE, your own agent) decides WHEN to search the documents, when to
walk the graph and when to inspect an entity. This server only retrieves: it returns cited evidence (document,
pages, section path, or ordered fact chains) and never answers on the model's behalf.

Configuration (environment):
  LAYOUT_RAG_INDEX     directory written by `layout-rag ingest`               (enables search_documents)
  LAYOUT_RAG_EMBEDDER  embedder used for that index (default: read from index.json)
  RAG_RERANKER         local cross-encoder for documents, or "none"          (default BAAI/bge-reranker-v2-m3)
  RAG_RERANK_POOL      candidates the cross-encoder scores when rerank=True   (default 20)
  GRAPH_WORKDIR        LightRAG working directory (graphwalk-rag)             (enables the graph tools)
  GRAPH_EMBED_MODEL    the model that embedded the graph: an OpenAI-compatible name, or local:<sentence-transformer>
  LLM_MODEL / LLM_BASE_URL / LLM_API_KEY   keyword extraction for graph_walk / guided_walk (OpenAI-compatible)

Run:  rag-mcp-server              (stdio)      rag-mcp-server --http 8765   (streamable HTTP)
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import cache
from pathlib import Path

try:                                    # mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("rag-tools")
_LATENCY: dict[str, list[float]] = {}


def _timed(name: str, t0: float) -> None:
    _LATENCY.setdefault(name, []).append(time.time() - t0)


# ------------------------------------------------------------------------------------------------ lazy backends
@cache
def _doc_retriever():
    from layout_rag import MemoryStore, Retriever
    from layout_rag.embed import Embedder

    index = os.environ.get("LAYOUT_RAG_INDEX")
    if not index:
        raise RuntimeError("search_documents is disabled: set LAYOUT_RAG_INDEX to a `layout-rag ingest` index")
    meta = Path(index) / "index.json"
    name = os.environ.get("LAYOUT_RAG_EMBEDDER") or (json.loads(meta.read_text())["embedder"] if meta.exists()
                                                    else "BAAI/bge-large-en-v1.5")
    rr_name = os.environ.get("RAG_RERANKER", "BAAI/bge-reranker-v2-m3")
    reranker = None
    if rr_name.lower() != "none":
        from layout_rag import CrossEncoderReranker

        reranker = CrossEncoderReranker(rr_name)
    if name == "hash":                  # deterministic test embedder: no model download
        from layout_rag.embed import HashEmbedder

        return Retriever(MemoryStore.load(index), HashEmbedder(), reranker=reranker)
    return Retriever(MemoryStore.load(index), Embedder(name), reranker=reranker)


@cache
def _graph():
    from graphwalk_rag import GraphStore

    wd = os.environ.get("GRAPH_WORKDIR")
    if not wd:
        raise RuntimeError("graph tools are disabled: set GRAPH_WORKDIR to a LightRAG working directory")
    return GraphStore(wd)


@cache
def _graph_models():
    from graphwalk_rag.llm import Chat, Embed, LocalReranker

    return (Chat(os.environ.get("LLM_MODEL", "gpt-4o-mini")),
            Embed(os.environ.get("GRAPH_EMBED_MODEL", "text-embedding-3-small")), LocalReranker())


# ------------------------------------------------------------------------------------------------ tools
@mcp.tool()
def search_documents(query: str, k: int = 5, mode: str = "hybrid", rerank: bool = False,
                     document: str | None = None) -> list[dict]:
    """Search the ingested PDF collection. Returns the top-k chunks with document name, 1-based pages, section
    path and text. mode: "hybrid" (dense + BM25, default), "dense" or "bm25". `document` restricts the search to
    one file name. rerank=True adds a local cross-encoder pass: more precise, much slower on CPU.
    Use this for questions answered by a passage, table row or definition."""
    t0 = time.time()
    hits = _doc_retriever().search(query, k=max(1, min(k, 20)), mode=mode, rerank=rerank,
                                   pool=int(os.environ.get("RAG_RERANK_POOL", "20")) if rerank else 50,
                                   filters={"doc_name": document} if document else None)
    _timed("search_documents", t0)
    return [{"document": h.chunk.doc_name, "pages": [p + 1 for p in h.chunk.pages],
             "section": " > ".join(h.chunk.heading_path), "score": round(h.score, 4), "text": h.chunk.body}
            for h in hits]


@mcp.tool()
def graph_walk(question: str, max_hops: int = 5, chains: int = 5) -> dict:
    """Multi-hop evidence from the knowledge graph: anchors the question's named entities, walks to plausible
    answers by bidirectional meet-in-the-middle, and returns the best ordered fact chains with their source
    documents. Use this when the answer is several relations away from what the question names."""
    from graphwalk_rag.paths import verbalize
    from graphwalk_rag.walk import GraphWalk, WalkConfig

    t0 = time.time()
    chat, embed, rr = _graph_models()
    r = GraphWalk(_graph(), chat, embed, rr, WalkConfig(max_hops=max(2, min(max_hops, 8)),
                                                        top_k=max(1, min(chains, 10)))).retrieve(question)
    _timed("graph_walk", t0)
    return {"keywords": r.keywords, "anchors": r.anchors, "candidate_paths": r.candidates,
            "chains": [verbalize(_graph(), p) for p in r.paths]}


@mcp.tool()
def guided_walk(question: str) -> dict:
    """Evidence from long structured corpora (regulations, policies, contracts): routes the question to the most
    relevant documents, walks question-relevant relations only, and returns fact chains plus the (document, page)
    evidence behind them."""
    from graphwalk_rag.guided import GuidedWalk
    from graphwalk_rag.paths import verbalize

    t0 = time.time()
    chat, embed, rr = _graph_models()
    r = GuidedWalk(_graph(), chat, embed, rr).walk(question)
    _timed("guided_walk", t0)
    return {"routed_documents": r["routed_docs"], "chains": [verbalize(_graph(), p) for p in r["paths"][:8]],
            "evidence_pages": [{"document": d, "page": p} for d, p in r["evidence_pages"][:20]]}


@mcp.tool()
def get_entity(name: str) -> dict:
    """Look up one knowledge-graph entity (exact or normalised name): type, description, degree, source docs."""
    t0 = time.time()
    gs = _graph()
    node = gs.resolve(name)
    _timed("get_entity", t0)
    if node is None:
        return {"found": False, "name": name}
    return {"found": True, "name": node, "type": gs.node_type(node), "description": gs.node_desc(node),
            "degree": gs.degree(node), "is_hub": gs.is_hub(node), "documents": sorted(gs.node_docs(node))[:10]}


@mcp.tool()
def get_neighbors(name: str, limit: int = 25) -> dict:
    """Relations of one entity: neighbour, relation description and source documents (hubs flagged)."""
    t0 = time.time()
    gs = _graph()
    node = gs.resolve(name)
    if node is None:
        return {"found": False, "name": name}
    out = []
    for nb in list(gs.neighbors(node))[: max(1, min(limit, 100))]:
        e = gs.edge(node, nb)
        out.append({"neighbor": nb, "relation": e["desc"][:300], "neighbor_is_hub": gs.is_hub(nb),
                    "documents": sorted({c["doc"] for c in gs.edge_chunks(node, nb)})[:3]})
    _timed("get_neighbors", t0)
    return {"found": True, "name": node, "degree": gs.degree(node), "neighbors": out}


@mcp.tool()
def server_stats() -> dict:
    """Which backends are configured, and per-tool latency (p50/p95 seconds) since the server started."""
    import statistics

    lat = {}
    for k, v in _LATENCY.items():
        s = sorted(v)
        lat[k] = {"calls": len(s), "p50": round(statistics.median(s), 3), "p95": round(s[int(0.95 * (len(s) - 1))], 3)}
    return {"documents_index": os.environ.get("LAYOUT_RAG_INDEX"), "graph_workdir": os.environ.get("GRAPH_WORKDIR"),
            "latency": lat}


def main() -> None:
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8765
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
