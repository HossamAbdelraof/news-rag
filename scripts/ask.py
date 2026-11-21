"""Search the index and print the results before and after reranking.

    uv run python -m scripts.ask "Why did OPEC cut oil output?"
    uv run python -m scripts.ask "Why did OPEC cut oil output?" --top-k 10 --candidates 50
    uv run python -m scripts.ask --interactive
"""

import argparse
import asyncio
import json
import sys
import textwrap

from scripts._common import setup_logging
from src.helpers.config import get_settings
from src.main import build_clients


def snippet(text: str, width: int = 110) -> str:
    return textwrap.shorten(text, width=width, placeholder=" ...")


def print_table(title: str, hits, reranked: bool) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for rank, hit in enumerate(hits, start=1):
        if reranked:
            move = hit.vector_rank - rank
            arrow = f"up {move}" if move > 0 else f"down {-move}" if move < 0 else "same"
            score = f"rerank {hit.rerank_score:+.3f} | was #{hit.vector_rank} ({arrow})"
        else:
            score = f"cosine {hit.vector_score:.3f}"
        category = hit.metadata.get("category") or "-"
        print(f"{rank:>2}. {hit.doc_id:<18} [{category}]  {score}")
        print(f"    {snippet(hit.text)}")


async def ask_one(controller, query: str, args) -> bool:
    defaults = get_settings().search
    ok, signal, _, result = await controller.search(
        query, candidates=args.candidates or defaults.SEARCH_CANDIDATES,
        top_k=args.top_k or defaults.SEARCH_TOP_K, rerank=not args.no_rerank, category=args.category,
    )
    if not ok:
        print(f"Error: {signal}")
        return False

    if args.json:
        for key in ("before_rerank", "after_rerank"):
            result[key] = [h.model_dump() for h in result[key]]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True

    print(f"\nQuery: {result['query']}")
    print(f"Embedding: {result['embedding_model']} | Reranker: {result['reranker_model'] or 'off'}")
    print(f"Candidates: {result['candidates']} chunks | Showing top {result['top_k']} documents | "
          f"Timings (ms): {result['timings']}")
    print_table("BEFORE reranking (vector search order)", result["before_rerank"], reranked=False)
    if result["reranker_model"]:
        print_table("AFTER reranking", result["after_rerank"], reranked=True)
    return True


async def main(args) -> int:
    clients = await build_clients()
    controller = clients.nlp_controller()
    if not args.interactive:
        if not args.query:
            print("Give a question, or use --interactive")
            return 1
        return 0 if await ask_one(controller, args.query, args) else 1

    if clients.reranker and not args.no_rerank:
        print(f"Loading reranker {clients.reranker.model_id} (once) ...")
        await clients.reranker.load()
    print("Ask questions (empty line or Ctrl+C to quit). Options from the command line apply to all.")
    while True:
        try:
            query = (await asyncio.to_thread(input, "\nQuestion> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        await ask_one(controller, query, args)
    return 0


if __name__ == "__main__":
    setup_logging(quiet=True)
    sys.stdout.reconfigure(encoding="utf-8")  # for Arabic on Windows
    sys.stdin.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="the question (omit with --interactive)")
    parser.add_argument("--interactive", "-i", action="store_true", help="load once, ask many questions")
    parser.add_argument("--candidates", type=int, help="chunks from vector search (default SEARCH_CANDIDATES)")
    parser.add_argument("--top-k", type=int, help="documents to show (default SEARCH_TOP_K)")
    parser.add_argument("--no-rerank", action="store_true", help="vector search only")
    parser.add_argument("--category", help="only search one R8 category, e.g. crude")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    sys.exit(asyncio.run(main(parser.parse_args())))
