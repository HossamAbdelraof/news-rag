"""Retrieval scores (nDCG@10, MRR@10, Recall) before and after reranking, on data/eval/testset.jsonl.

    uv run python -m scripts.evaluate --split final
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scripts._common import setup_logging
from src.helpers.config import PROJECT_ROOT, get_settings
from src.main import build_clients

DEFAULT_TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.jsonl"


# ---- metrics
def recall_at_k(ranked, relevant, k, required):
    return min(len(set(ranked[:k]) & relevant), required) / required


def mrr_at_k(ranked, relevant, k):
    return next((1.0 / r for r, d in enumerate(ranked[:k], 1) if d in relevant), 0.0)


def ndcg_at_k(ranked, relevant, k, required):
    dcg, credited = 0.0, 0
    for rank, doc in enumerate(ranked[:k], 1):
        if doc in relevant and credited < required:
            dcg += 1.0 / math.log2(rank + 1)
            credited += 1
    return dcg / sum(1.0 / math.log2(r + 1) for r in range(1, min(required, k) + 1))


def score(ranked: list[str], q: dict) -> dict:
    relevant = set(q["relevant_doc_ids"])
    required = min(len(q.get("source_doc_ids") or relevant), len(relevant))
    return {"ndcg@10": ndcg_at_k(ranked, relevant, 10, required), "mrr@10": mrr_at_k(ranked, relevant, 10),
            "recall@1": recall_at_k(ranked, relevant, 1, required),
            "recall@10": recall_at_k(ranked, relevant, 10, required)}


def to_testset_id(doc_id: str) -> str:
    return doc_id.removeprefix("r8-")  # "r8-test-00042" -> "test-00042"


async def main(args) -> int:
    questions = [json.loads(line) for line in Path(args.testset).read_text(encoding="utf-8").splitlines() if line]
    questions = [q for q in questions if q["qtype"] != "unanswerable" and q["split"] == args.split]
    clients = await build_clients()
    controller = clients.nlp_controller()
    settings = get_settings()
    print(f"collection {clients.collection} | chunks {settings.chunk.CHUNK_MAX_TOKENS} tokens, "
          f"overlap {settings.chunk.CHUNK_OVERLAP_TOKENS} | {len(questions)} {args.split} questions")

    rows, rerank_ms = [], []
    for i, q in enumerate(questions, 1):
        ok, signal, _, result = await controller.search(q["question"], candidates=args.candidates, top_k=10)
        if not ok:
            print(f"{q['qid']}: {signal}")
            return 1
        before = [to_testset_id(h.doc_id) for h in result["before_rerank"]]
        after = [to_testset_id(h.doc_id) for h in result["after_rerank"]]
        rows.append({"qid": q["qid"], "qtype": q["qtype"], "lang": q["lang"],
                     "dense": score(before, q), "reranked": score(after, q)})
        if "rerank_ms" in result["timings"] and i > 1:  # the first one is GPU warm-up
            rerank_ms.append(result["timings"]["rerank_ms"])
        if i % 40 == 0:
            print(f"  {i}/{len(questions)}")

    def mean(group, stage, metric):
        return round(statistics.mean(r[stage][metric] for r in group), 4)

    groups = {"all": rows}
    for key in ("qtype", "lang"):
        for value in sorted({r[key] for r in rows}):
            groups[f"{key}={value}"] = [r for r in rows if r[key] == value]

    summary = {"collection": clients.collection, "split": args.split, "candidates": args.candidates,
               "chunk_max_tokens": settings.chunk.CHUNK_MAX_TOKENS,
               "chunk_overlap_tokens": settings.chunk.CHUNK_OVERLAP_TOKENS,
               "median_rerank_ms": statistics.median(rerank_ms) if rerank_ms else None, "groups": {}}
    print(f"\n{'group':24s} {'n':>4s}  {'dense nDCG':>10s} {'MRR':>6s} {'R@10':>6s}  "
          f"{'rerank nDCG':>11s} {'MRR':>6s} {'R@10':>6s}")
    for name, group in groups.items():
        g = {stage: {m: mean(group, stage, m) for m in ("ndcg@10", "mrr@10", "recall@1", "recall@10")}
             for stage in ("dense", "reranked")}
        summary["groups"][name] = {"n": len(group), **g}
        d, r = g["dense"], g["reranked"]
        print(f"{name:24s} {len(group):4d}  {d['ndcg@10']:10.3f} {d['mrr@10']:6.3f} {d['recall@10']:6.3f}  "
              f"{r['ndcg@10']:11.3f} {r['mrr@10']:6.3f} {r['recall@10']:6.3f}")
    print(f"median rerank time: {summary['median_rerank_ms']} ms")

    out = PROJECT_ROOT / "data" / "eval_results" / f"{clients.collection}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_question": rows}, indent=1), encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    setup_logging(quiet=True)
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["tuning", "final"], default="tuning")
    parser.add_argument("--candidates", type=int, default=30, help="chunks sent to the reranker (benchmark: 30)")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    sys.exit(asyncio.run(main(parser.parse_args())))
