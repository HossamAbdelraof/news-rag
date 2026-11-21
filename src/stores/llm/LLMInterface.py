from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]] | None: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float] | None: ...
