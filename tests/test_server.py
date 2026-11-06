import asyncio
import base64
import hashlib
import json
import os
import sys

import networkx as nx
import numpy as np
import pytest
from layout_rag import MemoryStore
from layout_rag.embed import HashEmbedder
from layout_rag.types import Chunk

DIM = 8


def _vdb(path, rows, n):
    mat = np.eye(max(n, DIM), DIM, dtype=np.float32)[:n]
    path.write_text(json.dumps({"embedding_dim": DIM, "data": rows, "matrix": base64.b64encode(mat.tobytes()).decode()}))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    chunks = [Chunk(f"c{i}", "d", "policy.pdf", f"Policy > Section {i}\n\n{t}", t, ["Policy", f"Section {i}"], [i], 10)
              for i, t in enumerate(["Lenders must hold capital of at least ten percent.",
                                     "Branches report quarterly to the board.",
                                     "Customer complaints are resolved within thirty days."])]
    emb = HashEmbedder()
    st = MemoryStore()
    st.add(chunks, emb.embed_passages([c.text for c in chunks]))
    st.save(d / "index")
    (d / "index" / "index.json").write_text(json.dumps({"embedder": "hash"}))
    g = d / "graph"
    g.mkdir()
    G = nx.Graph()
    G.add_node("Lender", entity_type="org", description="A lending institution", source_id="k1")
    G.add_node("Capital Requirement", entity_type="rule", description="Minimum capital", source_id="k1")
    G.add_edge("Lender", "Capital Requirement", description="Lenders must meet the capital requirement",
               keywords="", source_id="k1", weight=1.0)
    nx.write_graphml(G, g / "graph_chunk_entity_relation.graphml")
    md = lambda s, p: p + hashlib.md5(s.encode()).hexdigest()
    _vdb(g / "vdb_entities.json", [{"__id__": md(n, "ent-"), "entity_name": n} for n in G.nodes], 2)
    _vdb(g / "vdb_relationships.json", [{"__id__": md("LenderCapital Requirement", "rel-")}], 1)
    (g / "kv_store_text_chunks.json").write_text(json.dumps(
        {"k1": {"content": "[Source: policy.pdf | Page 1]\nLenders...", "full_doc_id": "x", "chunk_order_index": 0}}))
    os.environ.update({"LAYOUT_RAG_INDEX": str(d / "index"), "RAG_RERANKER": "none", "GRAPH_WORKDIR": str(g)})
    return d


def test_search_documents(env):
    from rag_mcp_server import server

    hits = server.search_documents("capital lenders must hold", k=2, mode="bm25", rerank=False)
    assert hits[0]["document"] == "policy.pdf" and hits[0]["pages"] == [1] and "capital" in hits[0]["text"]
    assert hits[0]["section"] == "Policy > Section 0"


def test_entity_and_neighbors(env):
    from rag_mcp_server import server

    e = server.get_entity("lender")
    assert e["found"] and e["name"] == "Lender" and e["documents"] == ["policy.pdf"]
    nb = server.get_neighbors("Lender")
    assert nb["neighbors"][0]["neighbor"] == "Capital Requirement"
    assert nb["neighbors"][0]["documents"] == ["policy.pdf"]
    assert server.get_entity("nobody")["found"] is False
    assert "get_entity" in server.server_stats()["latency"]


def test_stdio_roundtrip(env):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "rag_mcp_server.server"], env=dict(os.environ))
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("get_entity", {"name": "Capital Requirement"})
            return names, res.content[0].text

    names, text = asyncio.run(run())
    assert {"search_documents", "graph_walk", "guided_walk", "get_entity", "get_neighbors", "server_stats"} <= names
    assert "Minimum capital" in text
