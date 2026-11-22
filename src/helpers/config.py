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
    # Qwen3 embeddings expect an instruction before the query (the documents were indexed without one)
    EMBEDDING_QUERY_PREFIX: str = ("Instruct: Given a question about financial news, retrieve Reuters news "
                                   "articles that answer it\nQuery: ")
    EMBEDDING_DOCUMENT_PREFIX: str = ""
    EMBEDDING_BATCH_SIZE: int = 64
    CHAT_API_KEY: str = ""  # used when EMBEDDING_API_KEY is empty (both are OpenRouter)

    @model_validator(mode="after")
    def use_chat_key(self):
        if not self.EMBEDDING_API_KEY:
            self.EMBEDDING_API_KEY = self.CHAT_API_KEY
        return self


class VectorDBConfig(BaseConfig):
    VECTOR_DB_PROVIDER: str = "chroma"  # chroma (read/write, local) | numpy (read-only export, for Vercel)
    VECTOR_DB_PATH: str = "chroma"      # chroma: folder in src/assets/database | numpy: folder in the project root
    VECTOR_DB_COLLECTION_PREFIX: str = "news"


class RerankerConfig(BaseConfig):
    RERANKER_ENABLED: bool = True
    RERANKER_PROVIDER: str = "openrouter"  # openrouter (API, no torch needed) | local (sentence-transformers)
    RERANKER_MODEL_ID: str = "qwen/qwen3-reranker-8b"
    RERANKER_API_URL: str = "https://openrouter.ai/api/v1"
    RERANKER_API_KEY: str = ""
    CHAT_API_KEY: str = ""  # used when RERANKER_API_KEY is empty (both are OpenRouter)
    RERANKER_DEVICE: str = "auto"  # auto | cuda | cpu
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_BATCH_SIZE: int = 16

    @model_validator(mode="after")
    def use_chat_key(self):
        if not self.RERANKER_API_KEY:
            self.RERANKER_API_KEY = self.CHAT_API_KEY
        return self


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
    # A full URL (Neon / Supabase / Vercel Postgres) wins over the parts below.
    POSTGRES_URL: str = ""
    POSTGRES_USERNAME: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_MAIN_DATABASE: str = "news_rag"

    @property
    def url(self) -> str:
        if self.POSTGRES_URL:
            return self._hosted_url()[0]
        return (f"postgresql+asyncpg://{self.POSTGRES_USERNAME}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_MAIN_DATABASE}")

    @property
    def connect_args(self) -> dict:
        return self._hosted_url()[1] if self.POSTGRES_URL else {}

    def _hosted_url(self) -> tuple[str, dict]:
        """postgres://...?sslmode=require -> asyncpg URL + connect args (asyncpg does not read sslmode)."""
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

        parts = urlsplit(self.POSTGRES_URL)
        query = dict(parse_qsl(parts.query))
        sslmode = query.pop("sslmode", None)
        query.pop("channel_binding", None)
        url = urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, urlencode(query), ""))
        # statement cache off: hosted poolers (PgBouncer) do not keep prepared statements
        args = {"statement_cache_size": 0}
        if sslmode and sslmode != "disable":
            args["ssl"] = "require"
        return url, args


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
