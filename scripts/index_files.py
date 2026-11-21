"""Read, chunk, embed and store files in Chroma.

    uv run python -m scripts.index_files ../datasets
    uv run python -m scripts.index_files ../datasets --reset
"""

import argparse
import asyncio
import json
import logging
import sys

from scripts._common import setup_logging, single_writer_lock
from src.controllers import BaseController, ProcessController
from src.helpers.config import get_settings
from src.main import build_clients

logger = logging.getLogger("uvicorn.error")


async def main(path: str, reset: bool) -> int:
    clients = await build_clients()
    logger.info("Embedding model: %s -> collection %s", clients.embedding.model_id, clients.collection)

    ok, signal, _, chunks, stats = ProcessController().process_path(path)
    logger.info("%s %s", signal, json.dumps(stats))
    if not ok:
        return 1

    db_folder = BaseController().get_database_path(get_settings().vectordb.VECTOR_DB_PATH)
    with single_writer_lock(db_folder):
        ok, signal, _, info = await clients.nlp_controller().index_chunks(chunks, reset=reset)
    logger.info("%s %s", signal, json.dumps(info))
    return 0 if ok else 1


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="file or folder to index")
    parser.add_argument("--reset", action="store_true", help="delete this model's collection first")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.path, args.reset)))
