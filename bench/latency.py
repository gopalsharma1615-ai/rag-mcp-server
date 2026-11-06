"""Per-tool latency on a real index and graph (no LLM calls): warm p50/p95 over a query list.

    LAYOUT_RAG_INDEX=index/ GRAPH_WORKDIR=graph/ python bench/latency.py --queries q.txt --entities e.txt
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from rag_mcp_server import server


def bench(fn, args_list):
    fn(**args_list[0])                       # warm-up: loads models and indexes
    t = []
    for a in args_list:
        t0 = time.time()
        fn(**a)
        t.append(time.time() - t0)
    t.sort()
    return {"calls": len(t), "p50_s": round(statistics.median(t), 3), "p95_s": round(t[int(0.95 * (len(t) - 1))], 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries")
    ap.add_argument("--entities")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    if a.queries:
        qs = [q for q in Path(a.queries).read_text().splitlines() if q.strip()]
        for rerank in (False, True):  # rerank uses RAG_RERANK_POOL candidates
            res[f"search_documents(hybrid, rerank={rerank})"] = bench(
                server.search_documents, [{"query": q, "k": 5, "rerank": rerank} for q in qs])
    if a.entities:
        es = [e for e in Path(a.entities).read_text().splitlines() if e.strip()]
        res["get_entity"] = bench(server.get_entity, [{"name": e} for e in es])
        res["get_neighbors"] = bench(server.get_neighbors, [{"name": e} for e in es])
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
