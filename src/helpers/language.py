import re

_ARABIC = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    """'ar' if the text contains Arabic letters, otherwise 'en'."""
    return "ar" if _ARABIC.search(text) else "en"
