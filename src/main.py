import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.controllers import BaseController, ChatController, NLPController
from src.helpers.config import Settings, get_settings
from src.models import ChatHistoryModel
from src.routes import base, chat, nlp
from src.stores.database import PostgresProvider
from src.stores.guardrails import GuardrailsProvider
from src.stores.llm import LLMInterface, LLMProviderFactory
from src.stores.monitoring import LangfuseProvider
from src.stores.reranker import RerankerInterface, RerankerProviderFactory
from src.stores.vectordb import VectorDBInterface, VectorDBProviderFactory

logger = logging.getLogger("uvicorn.error")


@dataclass
class Clients:
    settings: Settings
    embedding: LLMInterface
    vectordb: VectorDBInterface
    reranker: RerankerInterface | None
    collection: str
    monitoring: LangfuseProvider
    history: ChatHistoryModel | None = None  # None when Postgres is down
    db: PostgresProvider | None = None
    _chat: ChatController | None = field(default=None, repr=False)

    def nlp_controller(self) -> NLPController:
        return NLPController(self.embedding, self.vectordb, self.reranker, self.collection)

    def chat_controller(self) -> ChatController:
        # created on first use, so search works without a chat key
        if self._chat is None:
            s = self.settings
            chat_model = LLMProviderFactory(s.embedding, s.chat).create_chat_model()
            guards = GuardrailsProvider(s.embedding, s.chat, s.chat.CHAT_GUARD_THRESHOLD)
            self._chat = ChatController(self.nlp_controller(), chat_model, guards)
        return self._chat


async def build_clients() -> Clients:
    settings = get_settings()
    embedding = LLMProviderFactory(settings.embedding).create_embedding_client()
    vectordb_factory = VectorDBProviderFactory(settings.vectordb,
                                               BaseController().get_database_path(settings.vectordb.VECTOR_DB_PATH))
    vectordb = vectordb_factory.create()
    await vectordb.connect()
    clients = Clients(settings=settings, embedding=embedding, vectordb=vectordb,
                      reranker=RerankerProviderFactory(settings.reranker).create(),
                      collection=vectordb_factory.collection_name(embedding.model_id),
                      monitoring=LangfuseProvider(settings.langfuse))

    # Postgres is optional: without it only the saved chat threads are missing.
    db = PostgresProvider(settings.db.url)
    try:
        await db.connect()
        clients.db, clients.history = db, ChatHistoryModel(db.sessions)
        await clients.history.create_tables(db.engine)
    except Exception as e:
        logger.warning("Postgres not reachable (%s): chat threads are off", type(e).__name__)
    return clients


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.setLevel(settings.app.LOG_LEVEL)
    app.state.clients = await build_clients()
    logger.info("Embedding: %s | collection: %s | reranker: %s | chat: %s | langfuse: %s",
                app.state.clients.embedding.model_id, app.state.clients.collection,
                app.state.clients.reranker.model_id if app.state.clients.reranker else "off",
                settings.chat.CHAT_MODEL_ID, "on" if app.state.clients.monitoring.enabled else "off")
    yield
    app.state.clients.monitoring.flush()
    if app.state.clients.db:
        await app.state.clients.db.disconnect()


app = FastAPI(title="News RAG", lifespan=lifespan)
app.include_router(base.base_router)
app.include_router(nlp.nlp_router)
app.include_router(chat.chat_router)

# The chat page at http://localhost:8000/ (mounted last so the API routes come first)
app.mount("/", StaticFiles(directory=Path(__file__).parents[1] / "ui", html=True), name="ui")
