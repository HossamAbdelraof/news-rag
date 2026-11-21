from abc import ABC, abstractmethod


class RerankerInterface(ABC):
    """Scores (query, text) pairs; a higher score means more relevant."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def score(self, query: str, texts: list[str]) -> list[float] | None: ...
