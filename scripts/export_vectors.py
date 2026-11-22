"""Export the active Chroma collection to the read-only numpy store used on Vercel.

    uv run python -m scripts.export_vectors              # writes vectors/<collection>.npy + .jsonl
    uv run python -m scripts.export_vectors --out other_folder

Run it again after re-indexing. Needs the `local` extra (chromadb).
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from src.controllers import BaseController
from src.helpers.config import PROJECT_ROOT, get_settings
from src.stores.vectordb import VectorDBProviderFactory
from src.stores.vectordb.providers.NumpyProvider import NumpyProvider

BATCH = 1000


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="vectors", help="folder, relative to the project root")
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings

    settings = get_settings()
    collection = VectorDBProviderFactory(settings.vectordb, "").collection_name(settings.embedding.EMBEDDING_MODEL_ID)
    chroma_path = BaseController().get_database_path("chroma")
    col = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False)) \
        .get_collection(collection)

    total = col.count()
    ids, texts, metas, vectors = [], [], [], []
    for offset in range(0, total, BATCH):
        r = col.get(limit=BATCH, offset=offset, include=["documents", "metadatas", "embeddings"])
        ids += r["ids"]
        texts += r["documents"]
        metas += r["metadatas"]
        vectors.append(np.asarray(r["embeddings"], dtype=np.float32))
    matrix = np.concatenate(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{collection}.npy", matrix.astype(np.float16))
    with (out / f"{collection}.jsonl").open("w", encoding="utf-8") as f:
        for i, t, m in zip(ids, texts, metas):
            f.write(json.dumps({"id": i, "text": t, "metadata": m or {}}, ensure_ascii=False) + "\n")

    check = NumpyProvider(str(out))
    print(f"Exported {await check.count(collection)} chunks ({matrix.shape[1]} dims) of {collection} to {out}")


if __name__ == "__main__":
    asyncio.run(main())
