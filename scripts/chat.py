"""Chat in the terminal. Type "new" for a new conversation, an empty line to quit.

    uv run python -m scripts.chat
"""

import argparse
import asyncio
import sys
import textwrap
import uuid

from scripts._common import setup_logging
from src.main import build_clients


def print_result(r: dict, show_sources: int) -> None:
    print(f"\n{r['answer']}")
    if r["search_question"] != r["question"]:
        print(f"\n  (searched as: {r['search_question']})")
    if r["status"] == "answered":
        print("\n  Sources:")
        for s in r["sources"][:show_sources]:
            print(f"   [{s['n']}] {s['doc_id']} [{s['category']}] rerank {s['rerank_score']:+.3f}: "
                  f"{textwrap.shorten(s['text'], 90, placeholder=' ...')}")
    else:
        print(f"\n  ({r['status']}: {r['reason']})")
    timings = ", ".join(f"{k.removesuffix('_ms')} {v} ms" for k, v in r["timings"].items())
    print(f"  {timings}")


async def main(args) -> int:
    clients = await build_clients()
    try:
        chat = clients.chat_controller()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    if clients.reranker:
        print(f"Loading reranker {clients.reranker.model_id} (once) ...")
        await clients.reranker.load()
    monitoring = "on" if clients.monitoring.enabled else "off"
    print(f"Chat model: {clients.settings.chat.CHAT_MODEL_ID} | Langfuse: {monitoring}")
    print('Ask about the 1987 Reuters financial news. "new" = new conversation, empty line = quit.')

    history, session_id = [], str(uuid.uuid4())
    while True:
        try:
            question = (await asyncio.to_thread(input, "\nYou> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        if question.lower() == "new":
            history, session_id = [], str(uuid.uuid4())
            print("(new conversation)")
            continue
        result = await chat.answer(question, history, callbacks=clients.monitoring.callbacks(), session_id=session_id)
        print_result(result, args.show_sources)
        if result["status"] == "answered":
            history.append((question, result["answer"]))

    clients.monitoring.flush()
    return 0


if __name__ == "__main__":
    setup_logging(quiet=True)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-sources", type=int, default=3, help="how many sources to print")
    sys.exit(asyncio.run(main(parser.parse_args())))
