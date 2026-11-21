import asyncio
from types import SimpleNamespace

from src.controllers import ChatController
from src.models import SearchHit
from src.stores.guardrails import GuardrailsProvider, GuardResult


def _hit(doc_id, score, text="opec cut output to support prices"):
    return SearchHit(chunk_id=doc_id, doc_id=doc_id, text=text, metadata={"category": "crude"},
                     vector_rank=1, vector_score=0.8, rerank_score=score)


class FakeNLP:
    def __init__(self, score=0.95):
        self.score, self.queries = score, []
        self.reranker = SimpleNamespace(used_device="cpu")

    async def search(self, query, candidates, top_k, rerank=True, category=None):
        self.queries.append(query)
        return True, "ok", 200, {"after_rerank": [_hit("r8-test-00001", self.score), _hit("r8-test-00002", 0.2)]}


class FakeModel:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


class FakeGuards:
    def __init__(self, block_reason=None):
        self.block_reason = block_reason

    async def check_input(self, question):
        return GuardResult(False, self.block_reason) if self.block_reason else GuardResult(True)

    check_output = staticmethod(GuardrailsProvider.check_output)


def ask(nlp, model, guards, question, history=None):
    return asyncio.run(ChatController(nlp, model, guards).answer(question, history))


def test_answered_with_sources():
    model = FakeModel("OPEC cut output to support prices [1].")
    r = ask(FakeNLP(), model, FakeGuards(), "Why did OPEC cut output?")
    assert r["status"] == "answered" and r["answer"].endswith("[1].")
    assert [s["doc_id"] for s in r["sources"]] == ["r8-test-00001", "r8-test-00002"]
    assert "rewrite_ms" in r["timings"] and "generate_ms" in r["timings"]
    assert len(model.calls) == 1


def test_blocked_by_input_guard_in_arabic():
    model = FakeModel()
    r = ask(FakeNLP(), model, FakeGuards("advice"), "هل أشتري أسهم الآن؟")
    assert r["status"] == "blocked" and r["reason"] == "advice" and r["language"] == "ar"
    assert "نصائح استثمارية" in r["answer"] and model.calls == []


def test_low_relevance_never_calls_the_model():
    model = FakeModel()
    r = ask(FakeNLP(score=0.05), model, FakeGuards(), "Give me a banana bread recipe")
    assert r["status"] == "not_found" and r["reason"] == "low_relevance" and model.calls == []


def test_reranker_off_skips_the_relevance_gate():
    model = FakeModel("OPEC cut output to support prices [1].")
    r = asyncio.run(ChatController(FakeNLP(score=0.05), model, FakeGuards()).answer("Why did OPEC cut output?",
                                                                                     rerank=False))
    assert r["status"] == "answered" and r["reranked"] is False and r["top_score"] is None


def test_stream_yields_steps_and_final():
    async def collect():
        ctrl = ChatController(FakeNLP(), FakeModel("OPEC cut output to support prices [1]."), FakeGuards())
        return [e async for e in ctrl.stream("Why did OPEC cut output?")]

    events = asyncio.run(collect())
    steps = [e["name"] for e in events if e["type"] == "step"]
    assert steps == ["input_guard", "rewrite", "search", "relevance_gate", "generate", "output_guard"]
    assert events[-1]["type"] == "final" and events[-1]["status"] == "answered"
    assert {"input_guard_ms", "search_ms", "generate_ms", "output_guard_ms"} <= set(events[-1]["timings"])


def test_model_says_not_in_sources():
    r = ask(FakeNLP(), FakeModel("NOT_IN_SOURCES"), FakeGuards(), "Exactly how much did Primerica pay?")
    assert r["status"] == "not_found" and r["reason"] == "model_found_nothing"


def test_retry_once_then_safe_reply():
    model = FakeModel("OPEC cut output by 5 percent [1].", "OPEC cut output.")
    r = ask(FakeNLP(), model, FakeGuards(), "Why did OPEC cut output?")
    assert r["status"] == "unsafe" and r["reason"] == "no_citation" and len(model.calls) == 2


def test_retry_fixes_the_answer():
    model = FakeModel("OPEC cut output by 5 percent [1].", "OPEC cut output to support prices [1].")
    r = ask(FakeNLP(), model, FakeGuards(), "Why did OPEC cut output?")
    assert r["status"] == "answered" and len(model.calls) == 2


def test_follow_up_is_rewritten_before_search():
    nlp = FakeNLP()
    model = FakeModel("Why did Kuwait reject an emergency OPEC meeting?", "Kuwait saw no need for it [1].")
    r = ask(nlp, model, FakeGuards(), "and why did they say that?",
            history=[("What did Kuwait say about OPEC?", "Kuwait said no emergency meeting was planned [1].")])
    assert nlp.queries == ["Why did Kuwait reject an emergency OPEC meeting?"]
    assert r["search_question"] == nlp.queries[0] and r["status"] == "answered"
