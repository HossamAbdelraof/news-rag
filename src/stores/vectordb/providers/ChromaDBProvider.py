import asyncio

import chromadb
from chromadb.config import Settings

from ..VectorDBInterface import VectorDBInterface


class ChromaDBProvider(VectorDBInterface):
    """Local Chroma with cosine distance. Its client is not async, so calls run in a thread."""

    BATCH = 1000

    def __init__(self, path: str):
        self.path = path
        self.client = None

    async def connect(self) -> None:
        self.client = await asyncio.to_thread(
            chromadb.PersistentClient, path=self.path, settings=Settings(anonymized_telemetry=False)
        )

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 200}},
            embedding_function=None,
        )

    async def count(self, collection: str) -> int:
        return await asyncio.to_thread(lambda: self._collection(collection).count())

    async def existing_ids(self, collection: str, ids: list[str]) -> set[str]:
        def work():
            col, found = self._collection(collection), set()
            for i in range(0, len(ids), self.BATCH):
                found.update(col.get(ids=ids[i:i + self.BATCH], include=[])["ids"])
            return found
        return await asyncio.to_thread(work)

    async def insert(self, collection: str, ids, texts, vectors, metadatas) -> None:
        def work():
            col = self._collection(collection)
            for i in range(0, len(ids), self.BATCH):
                s = slice(i, i + self.BATCH)
                col.upsert(ids=ids[s], documents=texts[s], embeddings=vectors[s], metadatas=metadatas[s])
        await asyncio.to_thread(work)

    async def search(self, collection: str, vector: list[float], limit: int, where: dict | None = None) -> list[dict]:
        def work():
            return self._collection(collection).query(
                query_embeddings=[vector], n_results=limit, where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        r = await asyncio.to_thread(work)
        return [{"id": i, "text": t, "metadata": m or {}, "score": 1.0 - d}
                for i, t, m, d in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    async def delete_collection(self, collection: str) -> None:
        names = await self.list_collections()
        if collection in names:
            await asyncio.to_thread(self.client.delete_collection, collection)

    async def list_collections(self) -> dict[str, int]:
        def work():
            return {c.name: c.count() for c in self.client.list_collections()}
        return await asyncio.to_thread(work)
