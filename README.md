# rag-mcp-server

[![CI](https://github.com/gopalsharma1615-ai/rag-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/gopalsharma1615-ai/rag-mcp-server/actions/workflows/ci.yml)
![MCP](https://img.shields.io/badge/MCP-1.x%20%7C%202.x-blueviolet) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

**Your documents and knowledge graph as tools that any AI assistant can call**, over the
[Model Context Protocol](https://modelcontextprotocol.io).

The client (Claude Desktop, Claude Code, an IDE, or your own agent) decides *when* to search passages, when to
walk the graph, and when to inspect an entity. This server only retrieves. It returns cited evidence (document,
page, section, or ordered fact chains) and never answers on the model's behalf.

| tool | backed by | returns |
|---|---|---|
| `search_documents(query, k, mode, rerank, document)` | [layout-rag](https://github.com/gopalsharma1615-ai/layout-rag): hybrid dense + BM25, optional local cross-encoder | chunks with document, 1-based pages, section path |
| `graph_walk(question, max_hops, chains)` | [graphwalk-rag](https://github.com/gopalsharma1615-ai/graphwalk-rag): bidirectional meet-in-the-middle | anchors and the best ordered fact chains with sources |
| `guided_walk(question)` | graphwalk-rag: document-routed walk | routed documents, fact chains, (document, page) evidence |
| `get_entity(name)` / `get_neighbors(name)` | the graph store | type, description, degree, hub flag, relations with sources |
| `server_stats()` | this server | configured backends, per-tool p50/p95 latency |

## Run it

```bash
pip install "rag-mcp-server @ git+https://github.com/gopalsharma1615-ai/rag-mcp-server"

layout-rag ingest docs/*.pdf --index ./index          # documents (see layout-rag)
graphwalk-rag build docs/ --workdir ./graph           # graph (see graphwalk-rag; price it with --dry-run)

export LAYOUT_RAG_INDEX=./index GRAPH_WORKDIR=./graph
export GRAPH_EMBED_MODEL=text-embedding-3-small        # the model that embedded the graph
export LLM_MODEL=gpt-4o-mini                           # keyword extraction for the walks (or LLM_BASE_URL for vLLM)

claude mcp add rag-tools -- rag-mcp-server             # Claude Code (stdio)
rag-mcp-server --http 8765                             # or streamable HTTP for remote clients
```

Claude Desktop (`claude_desktop_config.json`):

```json
{"mcpServers": {"rag-tools": {"command": "rag-mcp-server",
  "env": {"LAYOUT_RAG_INDEX": "/path/index", "GRAPH_WORKDIR": "/path/graph"}}}}
```

Either backend is optional. Without `GRAPH_WORKDIR` only `search_documents` works, and vice versa. Indexes and
models load lazily on first use.

## Measured latency

Warm latency on a CPU-only VM (8 threads), per tool call, from `python bench/latency.py`:

| tool | setting | p50 | p95 |
|---|---|---|---|
| `search_documents` | hybrid, bge-large-en-v1.5, no rerank | **0.11 s** | 0.12 s |
| `search_documents` | + `bge-reranker-v2-m3` over 20 candidates | 39.7 s | 41.2 s |
| `search_documents` | + `bge-reranker-v2-m3` over 50 candidates | 58.6 s | 61.6 s |
| `get_entity` / `get_neighbors` | 52,474-node graph | < 1 ms | < 1 ms |

These measurements drove a design decision: **reranking is opt-in** (`rerank=False` by default, pool set by
`RAG_RERANK_POOL`, default 20). A 568M-parameter cross-encoder over 512-token chunks is impractical on CPU for an
interactive tool: even 20 candidates take about 40 s. Use a GPU, a smaller cross-encoder (`RAG_RERANKER`), or leave it off. The graph tools'
latency is dominated by the keyword-extraction LLM call and query embedding, not by the walk itself: the
traversal takes milliseconds (see graphwalk-rag).

## Evaluation plan

The agent-vs-pipeline comparison is planned and not yet run: an LLM client choosing these tools vs a fixed
single-pass pipeline on HotpotQA multi-hop, measuring answer accuracy, tool-selection accuracy, calls per
question, latency and $/question. It will live in
[research-agent](https://github.com/gopalsharma1615-ai/research-agent).

## Development

```bash
pip install -e ".[dev]" && pytest -q   # tool tests on a tiny index + graph, and a real stdio round-trip
```

MIT licence. Built by [Gopal Sharma](https://github.com/gopalsharma1615-ai).
