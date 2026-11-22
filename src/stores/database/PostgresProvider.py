import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

logger = logging.getLogger("uvicorn.error")


class PostgresProvider:
    def __init__(self, url: str, connect_args: dict | None = None):
        self.url = url
        self.connect_args = connect_args or {}
        self.engine: AsyncEngine | None = None
        self.sessions: async_sessionmaker | None = None

    async def connect(self) -> None:
        self.engine = create_async_engine(self.url, pool_pre_ping=True, connect_args=self.connect_args)
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        logger.info("Connected to Postgres")

    async def disconnect(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
