import logging
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.helpers.language import detect_language
from src.stores.llm.templates import prompts

from .BaseController import BaseController
from .NLPController import NLPController

logger = logging.getLogger("uvicorn.error")

STOP = {"blocked", "not_found", "error"}


def add_timings(old: dict | None, new: dict | None) -> dict:
    merged = dict(old or {})
    for name, ms in (new or {}).items():
        merged[name] = merged.get(name, 0) + ms
    return merged


class ChatState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    language: str
    search_question: str
    rerank: bool
    sources: list
    top_score: float | None
    answer: str
    status: str  # answered | blocked | not_found | unsafe | error | retry
    reason: str
    attempts: int
    timings: Annotated[dict, add_timings]


class ChatController(BaseController):
    def __init__(self, nlp: NLPController, chat_model, guards):
        super().__init__()
        self.nlp = nlp
        self.chat_model = chat_model
        self.guards = guards
        self.config = self.settings.chat
        self.graph = self.build_graph()

    def reply(self, state: ChatState, key: str) -> str:
        return prompts.REPLIES[state["language"]][key]

    # ---- steps

    async def input_guard(self, state: ChatState, config) -> ChatState:
        result = await self.guards.check_input(state["question"])
        if result.allowed:
            return {}
        return {"status": "blocked", "reason": result.reason, "answer": self.reply(state, result.reason)}

    async def rewrite(self, state: ChatState, config) -> ChatState:
        history = state.get("history") or []
        if not history:
            return {"search_question": state["question"]}

        turns = history[-self.config.CHAT_HISTORY_TURNS:]
        text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)
        messages = [
            SystemMessage(prompts.REWRITE_SYSTEM),
            HumanMessage(prompts.REWRITE_USER.format(history=text, question=state["question"])),
        ]
        reply = await self.chat_model.ainvoke(messages, config=config)
        return {"search_question": reply.content.strip() or state["question"]}

    async def search(self, state: ChatState, config) -> ChatState:
        ok, signal, _, result = await self.nlp.search(
            state["search_question"],
            candidates=self.config.CHAT_CANDIDATES,
            top_k=self.config.CHAT_TOP_K,
            rerank=state["rerank"],
        )
        if not ok:
            logger.error("Search failed: %s", signal)
            return {"status": "error", "reason": "search_failed", "answer": self.reply(state, "error")}

        hits = result["after_rerank"]
        top_score = hits[0].rerank_score if hits and state["rerank"] else None
        return {"sources": hits, "top_score": top_score}

    async def relevance_gate(self, state: ChatState, config) -> ChatState:
        if state.get("status") in STOP or not state["rerank"]:
            return {}
        score = state.get("top_score")
        if score is None or score < self.config.CHAT_RELEVANCE_THRESHOLD:
            return {"status": "not_found", "reason": "low_relevance", "answer": self.reply(state, "not_found")}
        return {}

    async def generate(self, state: ChatState, config) -> ChatState:
        sources = "\n".join(
            prompts.SOURCE_LINE.format(n=n, doc_id=h.doc_id, category=h.metadata.get("category") or "-", text=h.text)
            for n, h in enumerate(state["sources"], start=1)
        )
        language = prompts.LANGUAGE_NAMES[state["language"]]
        messages = [
            SystemMessage(prompts.ANSWER_SYSTEM.format(language=language)),
            HumanMessage(prompts.ANSWER_USER.format(sources=sources, question=state["question"])),
        ]
        reply = await self.chat_model.ainvoke(messages, config=config)
        answer = (reply.content or "").strip()

        if prompts.NOT_IN_SOURCES in answer:
            return {"status": "not_found", "reason": "model_found_nothing", "answer": self.reply(state, "not_found")}
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    async def output_guard(self, state: ChatState, config) -> ChatState:
        result = self.guards.check_output(state["answer"], len(state["sources"]))
        if result.allowed:
            return {"status": "answered"}

        logger.warning("Output guard: %s (attempt %d)", result.reason, state["attempts"])
        if state["attempts"] < 2:
            return {"status": "retry", "reason": result.reason}
        return {"status": "unsafe", "reason": result.reason, "answer": self.reply(state, "unsafe_answer")}

    # ---- graph

    @staticmethod
    def after_gate(state: ChatState) -> str:
        return END if state.get("status") in STOP else "generate"

    @staticmethod
    def after_generate(state: ChatState) -> str:
        return END if state.get("status") in STOP else "output_guard"

    @staticmethod
    def after_output_guard(state: ChatState) -> str:
        return "generate" if state.get("status") == "retry" else END

    def build_graph(self):
        graph = StateGraph(ChatState)
        for name in ("input_guard", "rewrite", "search", "relevance_gate", "generate", "output_guard"):
            graph.add_node(name, self.timed(name, getattr(self, name)))

        # The guard and rewrite -> search run in parallel, the gate waits for both.
        graph.add_edge(START, "input_guard")
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "search")
        graph.add_edge(["input_guard", "search"], "relevance_gate")
        graph.add_conditional_edges("relevance_gate", self.after_gate)
        graph.add_conditional_edges("generate", self.after_generate)
        graph.add_conditional_edges("output_guard", self.after_output_guard)
        return graph.compile()

    @staticmethod
    def timed(name, step):
        async def run(state: ChatState, config):
            started = time.perf_counter()
            update = await step(state, config)
            ms = round((time.perf_counter() - started) * 1000)
            return {**update, "timings": {f"{name}_ms": ms}}
        return run

    # ---- entry points

    def start(self, question, history, callbacks, session_id, rerank) -> tuple[dict, dict]:
        state = {
            "question": question,
            "history": history or [],
            "language": detect_language(question),
            "rerank": rerank and self.nlp.reranker is not None,
        }
        metadata = {"langfuse_session_id": session_id} if session_id else {}
        return state, {"callbacks": callbacks or [], "metadata": metadata}

    def failed(self, state: dict, e: Exception) -> dict:
        logger.exception("Chat failed: %s", e)
        return {**state, "status": "error", "reason": type(e).__name__, "answer": self.reply(state, "error")}

    async def answer(self, question: str, history=None, callbacks=None, session_id=None, rerank=True) -> dict:
        state, run_config = self.start(question, history, callbacks, session_id, rerank)
        try:
            final = await self.graph.ainvoke(state, config=run_config)
        except Exception as e:
            final = self.failed(state, e)
        return self.to_result(final)

    async def stream(self, question: str, history=None, callbacks=None, session_id=None, rerank=True):
        """Yields {"type": "step" | "token" | "restart" | "final", ...}. "final" is always last."""
        state, run_config = self.start(question, history, callbacks, session_id, rerank)
        final = dict(state)
        try:
            async for mode, chunk in self.graph.astream(state, config=run_config, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    message, meta = chunk
                    if meta.get("langgraph_node") == "generate" and message.content:
                        yield {"type": "token", "text": message.content}
                    continue

                for node, update in chunk.items():
                    update = dict(update or {})
                    timings = add_timings(final.get("timings"), update.pop("timings", None))
                    final.update(update, timings=timings)
                    yield {"type": "step", "name": node}
                    if node == "output_guard" and final.get("status") == "retry":
                        yield {"type": "restart"}
        except Exception as e:
            final = self.failed(state, e)
        yield {"type": "final", **self.to_result(final)}

    def to_result(self, final: dict) -> dict:
        reranked = final.get("rerank", False)
        sources = [
            {"n": n, "doc_id": h.doc_id, "category": h.metadata.get("category", ""),
             "rerank_score": h.rerank_score, "text": h.text}
            for n, h in enumerate(final.get("sources") or [], start=1)
        ]
        return {
            "question": final["question"],
            "search_question": final.get("search_question", final["question"]),
            "language": final["language"],
            "status": final.get("status"),
            "reason": final.get("reason", ""),
            "answer": final.get("answer", ""),
            "top_score": final.get("top_score"),
            "sources": sources,
            "model": self.config.CHAT_MODEL_ID,
            "timings": final.get("timings", {}),
            "reranked": reranked,
            "reranker_device": getattr(self.nlp.reranker, "used_device", None) if reranked else None,
        }
