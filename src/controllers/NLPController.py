import logging
import time

from fastapi import status

from src.models import Chunk, ResponseSignal, SearchHit
from src.stores.llm import LLMInterface
from src.stores.reranker import RerankerInterface
from src.stores.vectordb import VectorDBInterface

from .BaseController import BaseController

logger = logging.getLogger("uvicorn.error")


class NLPController(BaseController):
    def __init__(self, embedding_client: LLMInterface, vectordb_client: VectorDBInterface,
                 reranker: RerankerInterface | None, collection_name: str):
        super().__init__()
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client
        self.reranker = reranker
        self.collection = collection_name

    # ---- indexing
    async def index_chunks(self, chunks: list[Chunk], reset: bool = False,
                           batch_size: int = 256) -> tuple[bool, str, int, dict]:
        """Chunks that are already stored are skipped, so a stopped run can be started again."""
        if reset:
            await self.vectordb_client.delete_collection(self.collection)

        existing = await self.vectordb_client.existing_ids(self.collection, [c.id for c in chunks])
        todo = [c for c in chunks if c.id not in existing]
        started = time.perf_counter()
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            vectors = await self.embedding_client.embed_documents([c.text for c in batch])
            if vectors is None:
                return False, ResponseSignal.embedding_failed.value, status.HTTP_502_BAD_GATEWAY, {
                    "embedded_before_failure": i}
            await self.vectordb_client.insert(self.collection, [c.id for c in batch], [c.text for c in batch],
                                              vectors, [c.metadata() for c in batch])
            logger.info("embedded %d / %d chunks (%.0fs)", i + len(batch), len(todo), time.perf_counter() - started)

        return True, ResponseSignal.index_success.value, status.HTTP_200_OK, {
            "collection": self.collection, "embedding_model": self.embedding_client.model_id,
            "chunks_total": len(chunks), "already_indexed": len(existing), "newly_embedded": len(todo),
            "collection_size": await self.vectordb_client.count(self.collection),
            "seconds": round(time.perf_counter() - started, 1),
        }

    # ---- search
    @staticmethod
    def best_chunk_per_document(hits: list[SearchHit], limit: int) -> list[SearchHit]:
        """One chunk per document, so a long article can't fill the whole list."""
        seen, out = set(), []
        for hit in hits:
            if hit.doc_id not in seen:
                seen.add(hit.doc_id)
                out.append(hit)
            if len(out) == limit:
                break
        return out

    async def search(self, query: str, candidates: int, top_k: int, rerank: bool = True,
                     category: str | None = None) -> tuple[bool, str, int, dict]:
        """Vector search for `candidates` chunks, rerank them, keep `top_k` documents.
        Both lists (before and after reranking) are returned so they can be compared."""
        if await self.vectordb_client.count(self.collection) == 0:
            return False, ResponseSignal.collection_empty.value, status.HTTP_404_NOT_FOUND, {}
        timings = {}

        t = time.perf_counter()
        vector = await self.embedding_client.embed_query(query)
        timings["embed_ms"] = round((time.perf_counter() - t) * 1000)
        if vector is None:
            return False, ResponseSignal.embedding_failed.value, status.HTTP_502_BAD_GATEWAY, {}

        t = time.perf_counter()
        where = {"category": category} if category else None
        raw = await self.vectordb_client.search(self.collection, vector, limit=candidates, where=where)
        timings["vector_search_ms"] = round((time.perf_counter() - t) * 1000)
        hits = [SearchHit(chunk_id=r["id"], doc_id=r["metadata"].get("doc_id", ""), text=r["text"],
                          metadata=r["metadata"], vector_rank=rank, vector_score=round(r["score"], 4))
                for rank, r in enumerate(raw, start=1)]

        reranked = hits
        if rerank and self.reranker is not None and hits:
            if not self.reranker.is_loaded:  # first call only, timed on its own
                t = time.perf_counter()
                await self.reranker.load()
                timings["reranker_load_ms"] = round((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            scores = await self.reranker.score(query, [h.text for h in hits])
            timings["rerank_ms"] = round((time.perf_counter() - t) * 1000)
            if scores is None:
                return False, ResponseSignal.rerank_failed.value, status.HTTP_500_INTERNAL_SERVER_ERROR, {}
            for hit, score in zip(hits, scores):
                hit.rerank_score = round(score, 4)
            reranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)

        return True, ResponseSignal.search_success.value, status.HTTP_200_OK, {
            "query": query, "collection": self.collection,
            "embedding_model": self.embedding_client.model_id,
            "reranker_model": self.reranker.model_id if (rerank and self.reranker) else None,
            "candidates": len(hits), "top_k": top_k,
            "before_rerank": self.best_chunk_per_document(hits, top_k),
            "after_rerank": self.best_chunk_per_document(reranked, top_k),
            "timings": timings,
        }
