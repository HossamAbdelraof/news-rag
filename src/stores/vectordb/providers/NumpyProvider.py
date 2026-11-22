import asyncio
import json
from pathlib import Path

import numpy as np

from ..VectorDBInterface import VectorDBInterface


class NumpyProvider(VectorDBInterface):
    """Read-only vector store: exact cosine search over vectors exported from Chroma.

    Each collection is two files in `path`: <collection>.npy (unit-length float16 vectors, one row per
    chunk) and <collection>.jsonl (id, text and metadata per row, same order). Made by
    `python -m scripts.export_vectors`. Used where Chroma is too big or the disk is read-only (Vercel)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._collections: dict[str, tuple[np.ndarray, list[dict]]] = {}

    async def connect(self) -> None:
        if not self.path.is_dir():
            raise FileNotFoundError(f"Vector folder not found: {self.path}")

    def _files(self, collection: str) -> tuple[Path, Path]:
        return self.path / f"{collection}.npy", self.path / f"{collection}.jsonl"

    def _load(self, collection: str) -> tuple[np.ndarray, list[dict]] | None:
        if collection not in self._collections:
            vec_file, row_file = self._files(collection)
            if not vec_file.exists():
                return None
            vectors = np.load(vec_file).astype(np.float32)
            with row_file.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            if len(rows) != len(vectors):
                raise ValueError(f"{collection}: {len(vectors)} vectors but {len(rows)} rows")
            self._collections[collection] = (vectors, rows)
        return self._collections[collection]

    async def count(self, collection: str) -> int:
        loaded = await asyncio.to_thread(self._load, collection)
        return len(loaded[1]) if loaded else 0

    async def existing_ids(self, collection: str, ids: list[str]) -> set[str]:
        loaded = await asyncio.to_thread(self._load, collection)
        if not loaded:
            return set()
        wanted = set(ids)
        return {r["id"] for r in loaded[1] if r["id"] in wanted}

    async def insert(self, collection, ids, texts, vectors, metadatas) -> None:
        raise NotImplementedError("The numpy store is read-only: index with Chroma, then export_vectors")

    async def search(self, collection: str, vector: list[float], limit: int, where: dict | None = None) -> list[dict]:
        def work():
            loaded = self._load(collection)
            if not loaded:
                return []
            vectors, rows = loaded
            query = np.asarray(vector, dtype=np.float32)
            query /= np.linalg.norm(query) or 1.0
            scores = vectors @ query
            if where:
                keep = np.array([all(r["metadata"].get(k) == v for k, v in where.items()) for r in rows])
                scores = np.where(keep, scores, -np.inf)
            n = min(limit, int(np.isfinite(scores).sum()))
            if n == 0:
                return []
            top = np.argpartition(-scores, n - 1)[:n]
            top = top[np.argsort(-scores[top])]
            return [{"id": rows[i]["id"], "text": rows[i]["text"], "metadata": rows[i]["metadata"],
                     "score": float(scores[i])} for i in top]
        return await asyncio.to_thread(work)

    async def delete_collection(self, collection: str) -> None:
        raise NotImplementedError("The numpy store is read-only")

    async def list_collections(self) -> dict[str, int]:
        names = sorted(p.stem for p in self.path.glob("*.npy"))
        return {name: await self.count(name) for name in names}
