from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore")


class AppConfig(BaseConfig):
    APP_NAME: str = "news-rag"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"


class EmbeddingConfig(BaseConfig):
    EMBEDDING_API_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL_ID: str = "qwen/qwen3-embedding-4b"
    EMBEDDING_QUERY_PREFIX: str = ""
    EMBEDDING_DOCUMENT_PREFIX: str = ""
    EMBEDDING_BATCH_SIZE: int = 64
    CHAT_API_KEY: str = ""  # used when EMBEDDING_API_KEY is empty (both are OpenRouter)

    @model_validator(mode="after")
    def use_chat_key(self):
        if not self.EMBEDDING_API_KEY:
            self.EMBEDDING_API_KEY = self.CHAT_API_KEY
        return self


class VectorDBConfig(BaseConfig):
    VECTOR_DB_PATH: str = "chroma"
    VECTOR_DB_COLLECTION_PREFIX: str = "news"


class RerankerConfig(BaseConfig):
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "auto"  # auto | cuda | cpu
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_BATCH_SIZE: int = 16


class SearchConfig(BaseConfig):
    SEARCH_CANDIDATES: int = 30
    SEARCH_TOP_K: int = 5


class ChunkConfig(BaseConfig):
    CHUNK_TOKENIZER_ID: str = "BAAI/bge-reranker-v2-m3"
    CHUNK_MAX_TOKENS: int = 300
    CHUNK_OVERLAP_TOKENS: int = 50
    CHUNK_MIN_TAIL_TOKENS: int = 50

    @model_validator(mode="after")
    def check_overlap(self):
        if not 0 <= self.CHUNK_OVERLAP_TOKENS < self.CHUNK_MAX_TOKENS // 2:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be >= 0 and less than half of CHUNK_MAX_TOKENS")
        return self


class ChatConfig(BaseConfig):
    CHAT_API_URL: str = "https://openrouter.ai/api/v1"
    CHAT_API_KEY: str = ""
    CHAT_MODEL_ID: str = "google/gemma-4-26b-a4b-it"
    CHAT_TEMPERATURE: float = 0.1
    CHAT_MAX_TOKENS: int = 350
    CHAT_TOP_K: int = 5
    CHAT_CANDIDATES: int = 30
    CHAT_HISTORY_TURNS: int = 4
    CHAT_RELEVANCE_THRESHOLD: float = 0.1  # below this reranker score -> "not found in the sources"
    CHAT_GUARD_THRESHOLD: float = 0.6


class LangfuseConfig(BaseConfig):
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    @property
    def enabled(self) -> bool:
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)


class DBConfig(BaseConfig):
    POSTGRES_USERNAME: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_MAIN_DATABASE: str = "news_rag"

    @property
    def url(self) -> str:
        return (f"postgresql+asyncpg://{self.POSTGRES_USERNAME}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_MAIN_DATABASE}")


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    db: DBConfig = Field(default_factory=DBConfig)


@lru_cache
def get_settings() -> Settings:
    return Settings()
