import asyncio
import logging

from ..RerankerInterface import RerankerInterface

logger = logging.getLogger("uvicorn.error")


class CrossEncoderProvider(RerankerInterface):
    """sentence-transformers CrossEncoder, loaded on first use (fp16 on the GPU)."""

    def __init__(self, model_id: str, device: str = "cuda", batch_size: int = 16, max_length: int = 512):
        self._model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.used_device = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    async def load(self) -> None:
        if self.model is None:
            await asyncio.to_thread(self._load)

    def _load(self):
        import torch
        from sentence_transformers import CrossEncoder

        device = self.device
        if device in ("auto", "cuda"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.used_device = device
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = CrossEncoder(self._model_id, device=device, max_length=self.max_length, model_kwargs=kwargs)
        logger.info("Loaded reranker %s on %s", self._model_id, device)

    async def score(self, query: str, texts: list[str]) -> list[float] | None:
        if not texts:
            return []
        try:
            await self.load()
            scores = await asyncio.to_thread(
                self.model.predict, [[query, t] for t in texts], batch_size=self.batch_size, show_progress_bar=False
            )
            return [float(s) for s in scores]
        except Exception as e:
            logger.error("Reranking with %s failed: %s", self._model_id, e)
            return None
