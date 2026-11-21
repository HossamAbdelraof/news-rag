"""Input guard: NeMo Guardrails matches the question to example phrases (no LLM call).
Output guard: the answer must cite real sources and contain no numbers."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.helpers.config import ChatConfig, EmbeddingConfig

logger = logging.getLogger("uvicorn.error")

CONFIG_DIR = Path(__file__).parent / "config"
CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


class GuardrailsProvider:
    def __init__(self, embedding_config: EmbeddingConfig, chat_config: ChatConfig,
                 similarity_threshold: float = 0.6):
        self.embedding_config = embedding_config
        self.chat_config = chat_config
        self.similarity_threshold = similarity_threshold
        self._rails = None

    def _load(self):
        from nemoguardrails import LLMRails, RailsConfig

        yaml_text = (CONFIG_DIR / "config.yml").read_text(encoding="utf-8").format(
            chat_model_id=self.chat_config.CHAT_MODEL_ID, chat_api_url=self.chat_config.CHAT_API_URL,
            chat_api_key=self.chat_config.CHAT_API_KEY or "unused",
            embedding_model_id=self.embedding_config.EMBEDDING_MODEL_ID,
            embedding_api_url=self.embedding_config.EMBEDDING_API_URL,
            embedding_api_key=self.embedding_config.EMBEDDING_API_KEY,
            similarity_threshold=self.similarity_threshold,
        )
        colang_text = (CONFIG_DIR / "rails.co").read_text(encoding="utf-8")
        self._rails = LLMRails(RailsConfig.from_content(yaml_content=yaml_text, colang_content=colang_text))

    async def check_input(self, question: str) -> GuardResult:
        if self._rails is None:
            self._load()
        response = await self._rails.generate_async(messages=[{"role": "user", "content": question}])
        decision = str(response.get("content", "")).strip()
        if decision.startswith("BLOCK:"):
            return GuardResult(allowed=False, reason=decision.removeprefix("BLOCK:"))
        return GuardResult(allowed=True)

    @staticmethod
    def check_output(answer: str, n_sources: int) -> GuardResult:
        if not answer.strip():
            return GuardResult(False, "empty_answer")
        cited = [int(n) for n in CITATION.findall(answer)]
        if not cited:
            return GuardResult(False, "no_citation")
        if any(n < 1 or n > n_sources for n in cited):
            return GuardResult(False, "bad_citation")
        # only citations and the corpus year may contain digits
        if re.search(r"\d", CITATION.sub("", answer).replace("1987", "")):
            return GuardResult(False, "invented_numbers")
        return GuardResult(True)
