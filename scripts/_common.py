import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import psutil

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")


def setup_logging(quiet: bool = False) -> None:
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(levelname)s  %(message)s")
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING if quiet else logging.INFO)
    for noisy in ("httpx", "httpcore", "openai", "transformers", "docling", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


@contextmanager
def single_writer_lock(db_folder: str):
    """Only one indexing run may write to Chroma at a time (two at once broke the index)."""
    lock = Path(db_folder) / ".index.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
        except ValueError:
            other = -1
        if other > 0 and other != os.getpid() and psutil.pid_exists(other):
            raise SystemExit(f"Another indexing run (PID {other}) is writing to {db_folder}. "
                             f"Wait for it to finish, or delete {lock} if that process is gone.")
        lock.unlink(missing_ok=True)  # left by a crashed run
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
