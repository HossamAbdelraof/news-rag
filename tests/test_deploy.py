"""Pieces used on Vercel: the read-only numpy vector store and hosted Postgres URLs."""

import asyncio
import json

import numpy as np

from src.helpers.config import DBConfig
from src.stores.vectordb.providers.NumpyProvider import NumpyProvider


def write_store(folder, name="news_test"):
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]], dtype=np.float16)
    np.save(folder / f"{name}.npy", vectors)
    rows = [{"id": "a", "text": "oil", "metadata": {"category": "crude"}},
            {"id": "b", "text": "grain", "metadata": {"category": "grain"}},
            {"id": "c", "text": "oil and grain", "metadata": {"category": "crude"}}]
    (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return name


def test_numpy_store_search_and_filter(tmp_path):
    name = write_store(tmp_path)
    store = NumpyProvider(str(tmp_path))

    async def run():
        await store.connect()
        assert await store.count(name) == 3
        assert await store.count("missing") == 0
        hits = await store.search(name, [2, 0, 0], limit=2)
        assert [h["id"] for h in hits] == ["a", "c"]
        assert abs(hits[0]["score"] - 1.0) < 1e-3
        filtered = await store.search(name, [0, 1, 0], limit=5, where={"category": "crude"})
        assert [h["id"] for h in filtered] == ["c", "a"]
        assert await store.list_collections() == {name: 3}

    asyncio.run(run())


def test_hosted_postgres_url():
    config = DBConfig(POSTGRES_URL="postgres://u:p@ep-x.neon.tech/news_rag?sslmode=require&channel_binding=require")
    assert config.url == "postgresql+asyncpg://u:p@ep-x.neon.tech/news_rag"
    assert config.connect_args == {"statement_cache_size": 0, "ssl": "require"}


def test_postgres_parts_without_url():
    config = DBConfig(POSTGRES_URL="", POSTGRES_HOST="db", POSTGRES_PASSWORD="pw")
    assert config.url.startswith("postgresql+asyncpg://") and "@db:" in config.url
    assert config.connect_args == {}


def test_in_memory_history():
    from src.models import InMemoryChatHistory

    async def run():
        h = InMemoryChatHistory()
        t = await h.create_thread()
        await h.add_turn(t["id"], "Why did OPEC cut output?", "To defend prices [1].", {"status": "answered"})
        await h.add_turn(t["id"], "Pizza?", "Not found.", {"status": "not_found"})
        assert (await h.get_thread(t["id"]))["title"] == "Why did OPEC cut output?"
        assert await h.get_history(t["id"]) == [("Why did OPEC cut output?", "To defend prices [1].")]
        assert (await h.list_threads())[0]["questions"] == 2
        assert await h.delete_thread(t["id"]) and await h.get_thread(t["id"]) is None

    asyncio.run(run())
