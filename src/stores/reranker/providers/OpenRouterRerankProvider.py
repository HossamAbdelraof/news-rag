import logging

import httpx

from ..RerankerInterface import RerankerInterface

logger = logging.getLogger("uvicorn.error")


class OpenRouterRerankProvider(RerankerInterface):
    """Reranker over the OpenRouter /rerank API (no local model, so it runs on serverless hosts)."""

    used_device = "api"

    def __init__(self, model_id: str, api_url: str, api_key: str, timeout: float = 30.0):
        self._model_id = model_id
        self.url = api_url.rstrip("/") + "/rerank"
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    async def load(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout,
                                             headers={"Authorization": f"Bearer {self.api_key}"})

    async def score(self, query: str, texts: list[str]) -> list[float] | None:
        if not texts:
            return []
        try:
            await self.load()
            r = await self._client.post(self.url, json={"model": self._model_id, "query": query,
                                                        "documents": texts, "top_n": len(texts)})
            r.raise_for_status()
            scores = [0.0] * len(texts)
            for item in r.json()["results"]:
                scores[item["index"]] = float(item["relevance_score"])
            return scores
        except Exception as e:
            logger.error("Reranking with %s failed: %s", self._model_id, e)
            return None
