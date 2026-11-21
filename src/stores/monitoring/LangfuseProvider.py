import logging

from src.helpers.config import LangfuseConfig

logger = logging.getLogger("uvicorn.error")


class LangfuseProvider:
    """Sends traces to Langfuse when its keys are set, otherwise does nothing."""

    def __init__(self, config: LangfuseConfig):
        self.config = config
        self.client = None
        if config.enabled:
            from langfuse import Langfuse

            self.client = Langfuse(public_key=config.LANGFUSE_PUBLIC_KEY, secret_key=config.LANGFUSE_SECRET_KEY,
                                   host=config.LANGFUSE_HOST)
            logger.info("Langfuse tracing on (%s)", config.LANGFUSE_HOST)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def callbacks(self) -> list:
        if not self.enabled:
            return []
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler(public_key=self.config.LANGFUSE_PUBLIC_KEY)]

    def flush(self) -> None:
        if self.enabled:
            self.client.flush()
