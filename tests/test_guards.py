import asyncio
import urllib.request

import pytest

from src.helpers.config import get_settings
from src.stores.guardrails import GuardrailsProvider

check = GuardrailsProvider.check_output


def test_output_rules():
    assert check("OPEC cut output to support prices [1].", n_sources=3).allowed
    assert check("Icahn withdrew the bid [1][2].", n_sources=2).allowed
    assert check("", 3).reason == "empty_answer"
    assert check("OPEC cut output.", 3).reason == "no_citation"
    assert check("OPEC cut output [4].", 3).reason == "bad_citation"
    assert check("OPEC cut output by 5 percent [1].", 3).reason == "invented_numbers"
    assert check("The sources are from 1987: OPEC cut output [1].", 3).allowed
    assert check("In 1988 OPEC cut output [1].", 3).reason == "invented_numbers"


def embedding_api_up() -> bool:
    if not get_settings().embedding.EMBEDDING_API_KEY:
        return False
    try:
        urllib.request.urlopen(get_settings().embedding.EMBEDDING_API_URL.rstrip("/") + "/models", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not embedding_api_up(), reason="no embedding API key, or the API is not reachable")
def test_input_guard_live():
    s = get_settings()
    guard = GuardrailsProvider(s.embedding, s.chat, s.chat.CHAT_GUARD_THRESHOLD)

    async def run():
        return [await guard.check_input(q) for q in (
            "Give me a recipe for banana bread",
            "Should I buy Texaco shares now?",
            "Why did OPEC cut oil output?",
            "ماذا قال بنك اليابان عن أسعار الفائدة؟",
        )]

    recipe, advice, news, arabic_news = asyncio.run(run())
    # off-topic may pass here, the relevance gate stops it later
    assert not advice.allowed and advice.reason == "advice"
    assert news.allowed and arabic_news.allowed
    assert recipe.allowed or recipe.reason == "off_topic"
