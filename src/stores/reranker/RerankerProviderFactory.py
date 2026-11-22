from src.helpers.config import RerankerConfig

from .providers import CrossEncoderProvider, OpenRouterRerankProvider
from .RerankerInterface import RerankerInterface


class RerankerProviderFactory:
    def __init__(self, config: RerankerConfig):
        self.config = config

    def create(self) -> RerankerInterface | None:
        if not self.config.RERANKER_ENABLED:
            return None
        if self.config.RERANKER_PROVIDER == "openrouter":
            return OpenRouterRerankProvider(model_id=self.config.RERANKER_MODEL_ID,
                                            api_url=self.config.RERANKER_API_URL,
                                            api_key=self.config.RERANKER_API_KEY)
        return CrossEncoderProvider(model_id=self.config.RERANKER_MODEL_ID, device=self.config.RERANKER_DEVICE,
                                    batch_size=self.config.RERANKER_BATCH_SIZE,
                                    max_length=self.config.RERANKER_MAX_LENGTH)
