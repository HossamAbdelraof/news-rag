import asyncio

from src.controllers import NLPController, ProcessController
from src.helpers.config import VectorDBConfig
from src.stores.vectordb import VectorDBProviderFactory

run = asyncio.run


def test_r8_detection_cleaning_and_duplicates(tmp_path):
    r8 = tmp_path / "r8-test-all-terms.txt"
    r8.write_text("crude\topec cuts output reuter\nearn\tprofit rose reuter\ncrude\topec cuts output reuter\n",
                  encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Oil\n\nThe oil market was calm.", encoding="utf-8")
    assert ProcessController.is_r8_file(r8) and not ProcessController.is_r8_file(tmp_path / "notes.md")

    ok, _, _, chunks, stats = ProcessController().process_path(str(tmp_path))
    assert ok and stats["r8_articles"] == 2 and stats["r8_duplicates_skipped"] == 1 and stats["documents"] == 1
    r8_chunks = [c for c in chunks if c.source_file == r8.name]
    assert [c.doc_id for c in r8_chunks] == ["r8-test-00001", "r8-test-00002"]
    assert r8_chunks[0].text == "opec cuts output" and r8_chunks[0].category == "crude"
    assert any(c.source_file == "notes.md" and "oil market" in c.text for c in chunks)


def test_long_article_chunks_overlap_and_stay_within_limit(tmp_path):
    vocab = ["oil", "price", "bank", "rate", "trade", "stock", "market", "rise"]
    words = [vocab[i % len(vocab)] + f" w{i}" for i in range(1500)]
    (tmp_path / "r8-test-long.txt").write_text("crude\t" + " ".join(words) + "\n", encoding="utf-8")

    proc = ProcessController()
    ok, _, _, chunks, _ = proc.process_path(str(tmp_path))
    limit, overlap = proc.config.CHUNK_MAX_TOKENS, proc.config.CHUNK_OVERLAP_TOKENS
    assert ok and len(chunks) > 2
    assert all(proc.count_tokens(c.text) <= limit for c in chunks)
    assert [c.chunk_order for c in chunks] == list(range(1, len(chunks) + 1))
    for prev, cur in zip(chunks, chunks[1:]):
        repeated = proc.overlap_tail(prev.text)
        assert repeated and cur.text.startswith(repeated)
        assert 0 < proc.count_tokens(repeated) <= overlap


def test_overlap_off(tmp_path, monkeypatch):
    proc = ProcessController()
    monkeypatch.setattr(proc.config, "CHUNK_OVERLAP_TOKENS", 0)
    assert proc.overlap_tail("some words here") == ""


def test_missing_path():
    ok, signal, code, _, _ = ProcessController().process_path("does/not/exist")
    assert not ok and code == 404


def test_collection_name_per_model():
    f = VectorDBProviderFactory(VectorDBConfig(), "unused")
    assert f.collection_name("text-embedding-qwen3-embedding-4b") == "news_qwen3-embedding-4b"
    assert f.collection_name("text-embedding-qwen3-embedding-0.6b") == "news_qwen3-embedding-0.6b"


class FakeEmbedder:
    model_id = "fake"
    calls = 0

    async def embed_documents(self, texts):
        FakeEmbedder.calls += len(texts)
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        v = [float(text.count(w)) + 0.01 for w in ("oil", "opec", "bank", "rate")]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


class FakeReranker:
    model_id = "fake-reranker"
    is_loaded = True

    async def load(self):
        pass

    async def score(self, query, texts):
        return [1.0 if "bank" in t else 0.0 for t in texts]


def test_index_resume_and_search_before_after(tmp_path):
    from src.models import Chunk

    vdb = VectorDBProviderFactory(VectorDBConfig(), str(tmp_path)).create()
    run(vdb.connect())
    ctrl = NLPController(FakeEmbedder(), vdb, FakeReranker(), "news_test")
    chunks = [Chunk(doc_id="a", chunk_order=1, text="oil opec oil", source_file="f"),
              Chunk(doc_id="a", chunk_order=2, text="oil opec more", source_file="f"),
              Chunk(doc_id="b", chunk_order=1, text="bank rate oil", source_file="f"),
              Chunk(doc_id="c", chunk_order=1, text="bank rate", source_file="f")]
    FakeEmbedder.calls = 0
    ok, _, _, info = run(ctrl.index_chunks(chunks))
    assert ok and info["newly_embedded"] == 4
    ok, _, _, info = run(ctrl.index_chunks(chunks))  # nothing new the second time
    assert info["already_indexed"] == 4 and info["newly_embedded"] == 0 and FakeEmbedder.calls == 4

    ok, _, _, result = run(ctrl.search("oil opec", candidates=4, top_k=2))
    assert ok and result["candidates"] == 4
    before = [h.doc_id for h in result["before_rerank"]]
    after = [h.doc_id for h in result["after_rerank"]]
    assert before[0] == "a" and len(before) == 2 and len(set(before)) == 2
    assert after[0] in {"b", "c"} and result["after_rerank"][0].vector_rank > 1

    ok, _, _, no_rerank = run(ctrl.search("oil opec", candidates=4, top_k=3, rerank=False))
    assert no_rerank["reranker_model"] is None and no_rerank["after_rerank"] == no_rerank["before_rerank"]
