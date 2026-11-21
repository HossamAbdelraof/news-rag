import logging

from openai import AsyncOpenAI

from ..LLMInterface import LLMInterface

logger = logging.getLogger("uvicorn.error")


class OpenAIProvider(LLMInterface):
    """Embeddings from any OpenAI-compatible server (OpenRouter or LM Studio). Returns None on failure."""

    def __init__(self, api_url: str, api_key: str, model_id: str, query_prefix: str = "",
                 document_prefix: str = "", batch_size: int = 64):
        self.client = AsyncOpenAI(base_url=api_url, api_key=api_key, timeout=300)
        self._model_id = model_id
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size

    @property
    def model_id(self) -> str:
        return self._model_id

    async def _embed(self, inputs: list[str]) -> list[list[float]] | None:
        vectors = []
        try:
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i + self.batch_size]
                response = await self.client.embeddings.create(model=self._model_id, input=batch)
                vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        except Exception as e:
            logger.error("Embedding with %s failed: %s", self._model_id, e)
            return None
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        return await self._embed([self.document_prefix + t for t in texts])

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self._embed([self.query_prefix + text])
        return vectors[0] if vectors else None
