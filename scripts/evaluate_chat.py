"""Chat checks.

    uv run python -m scripts.evaluate_chat gate    # top reranker scores, to pick CHAT_RELEVANCE_THRESHOLD
    uv run python -m scripts.evaluate_chat guard   # input guard on held-out questions
    uv run python -m scripts.evaluate_chat chat    # the full chat (needs CHAT_API_KEY)
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

from scripts._common import setup_logging
from src.helpers.config import PROJECT_ROOT
from src.main import build_clients
from src.stores.guardrails import GuardrailsProvider

TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.jsonl"
HELDOUT = PROJECT_ROOT / "data" / "eval" / "guard_heldout.jsonl"
OUT = PROJECT_ROOT / "data" / "eval_results"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_of(q: dict) -> str:
    return f"unanswerable:{q['category']}" if q["qtype"] == "unanswerable" else "answerable"


def ids(doc_ids) -> set[str]:
    return {d.removeprefix("r8-") for d in doc_ids}


async def gate(clients, args) -> dict:
    nlp = clients.nlp_controller()
    scores = defaultdict(list)
    for q in read_jsonl(TESTSET):
        _, _, _, r = await nlp.search(q["question"], candidates=30, top_k=5)
        top = r["after_rerank"][0].rerank_score if r["after_rerank"] else 0.0
        if q["qtype"] != "unanswerable":
            found = bool(ids(h.doc_id for h in r["after_rerank"]) & set(q["relevant_doc_ids"]))
            scores["answerable, correct article retrieved" if found else "answerable, correct article missed"].append(top)
        else:
            scores[group_of(q)].append(top)
    print(f"\n{'group':42s} {'n':>4s} {'min':>7s} {'median':>7s} {'max':>7s}")
    for name, s in sorted(scores.items()):
        print(f"{name:42s} {len(s):4d} {min(s):7.3f} {statistics.median(s):7.3f} {max(s):7.3f}")
    print("\nthreshold  answerable(retrieved) kept   out-of-period/off-topic stopped")
    good = scores["answerable, correct article retrieved"]
    outside = scores.get("unanswerable:out_of_period", []) + scores.get("unanswerable:out_of_domain", [])
    for th in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        print(f"  {th:5.2f}      {sum(s >= th for s in good):3d}/{len(good)}"
              f"                     {sum(s < th for s in outside):2d}/{len(outside)}")
    return {k: v for k, v in scores.items()}


async def guard(clients, args) -> dict:
    s = clients.settings
    g = GuardrailsProvider(s.embedding, s.chat, s.chat.CHAT_GUARD_THRESHOLD)
    rows = []
    for q in read_jsonl(HELDOUT):
        r = await g.check_input(q["question"])
        rows.append({**q, "got": "allow" if r.allowed else "block", "reason": r.reason})
    print(f"\nInput guard on {len(rows)} held-out questions (threshold {s.chat.CHAT_GUARD_THRESHOLD}):")
    for grp in ("advice", "off_topic", "news_tricky", "news_plain"):
        part = [r for r in rows if r["group"] == grp]
        correct = sum(r["got"] == r["expected"] for r in part)
        print(f"  {grp:12s} expected {part[0]['expected']:5s}: correct {correct}/{len(part)}")
        for r in part:
            if r["got"] != r["expected"]:
                print(f"      wrong ({r['got']}): [{r['lang']}] {r['question']}")
    return {"rows": rows}


async def chat(clients, args) -> dict:
    controller = clients.chat_controller()
    questions = read_jsonl(TESTSET)
    unanswerable = [q for q in questions if q["qtype"] == "unanswerable"]
    answerable = [q for q in questions if q["qtype"] != "unanswerable" and q["split"] == "tuning"]
    random.Random(7).shuffle(answerable)
    sample = unanswerable + answerable[:args.sample]

    rows = []
    for i, q in enumerate(sample, 1):
        r = await controller.answer(q["question"], callbacks=clients.monitoring.callbacks(),
                                    session_id=f"eval-{q['qid']}")
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", r["answer"])]
        cited = {r["sources"][n - 1]["doc_id"].removeprefix("r8-") for n in numbers if 0 < n <= len(r["sources"])}
        rows.append({"qid": q["qid"], "group": group_of(q), "lang": q["lang"], "status": r["status"],
                     "reason": r["reason"], "cites_correct": bool(cited & set(q["relevant_doc_ids"])),
                     "answer": r["answer"], "timings": r["timings"]})
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    print("\nUnanswerable questions (should not get a normal answer):")
    for grp in sorted({r["group"] for r in rows if r["group"] != "answerable"}):
        part = [r for r in rows if r["group"] == grp]
        print(f"  {grp:30s} {dict(Counter(r['status'] for r in part))}")
    ans = [r for r in rows if r["group"] == "answerable"]
    answered = [r for r in ans if r["status"] == "answered"]
    print(f"\nAnswerable sample ({len(ans)}): statuses {dict(Counter(r['status'] for r in ans))}")
    if answered:
        print(f"  answered and citing a correct article: {sum(r['cites_correct'] for r in answered)}/{len(answered)}")
    gen = [r["timings"].get("generate_ms", 0) for r in answered]
    if gen:
        print(f"  median generation time: {statistics.median(gen):.0f} ms")
    clients.monitoring.flush()
    return {"rows": rows}


async def main(args) -> int:
    clients = await build_clients()
    if clients.reranker:
        await clients.reranker.load()
    OUT.mkdir(parents=True, exist_ok=True)
    result = await {"gate": gate, "guard": guard, "chat": chat}[args.mode](clients, args)
    out = OUT / f"chat_eval_{args.mode}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    setup_logging(quiet=True)
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["gate", "guard", "chat"])
    parser.add_argument("--sample", type=int, default=40, help="answerable questions in 'chat' mode")
    sys.exit(asyncio.run(main(parser.parse_args())))
