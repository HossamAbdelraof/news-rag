from langchain_openai import ChatOpenAI

from src.helpers.config import ChatConfig, EmbeddingConfig

from .LLMInterface import LLMInterface
from .providers import OpenAIProvider


class LLMProviderFactory:
    def __init__(self, embedding_config: EmbeddingConfig, chat_config: ChatConfig | None = None):
        self.embedding_config = embedding_config
        self.chat_config = chat_config

    def create_embedding_client(self) -> LLMInterface:
        c = self.embedding_config
        return OpenAIProvider(
            api_url=c.EMBEDDING_API_URL,
            api_key=c.EMBEDDING_API_KEY,
            model_id=c.EMBEDDING_MODEL_ID,
            query_prefix=c.EMBEDDING_QUERY_PREFIX,
            document_prefix=c.EMBEDDING_DOCUMENT_PREFIX,
            batch_size=c.EMBEDDING_BATCH_SIZE,
        )

    def create_chat_model(self) -> ChatOpenAI:
        c = self.chat_config
        if c is None or not c.CHAT_API_KEY:
            raise ValueError("CHAT_API_KEY is empty: add your OpenRouter key to .env")
        return ChatOpenAI(
            base_url=c.CHAT_API_URL,
            api_key=c.CHAT_API_KEY,
            model=c.CHAT_MODEL_ID,
            temperature=c.CHAT_TEMPERATURE,
            max_tokens=c.CHAT_MAX_TOKENS,
            timeout=60,
            max_retries=2,
        )
