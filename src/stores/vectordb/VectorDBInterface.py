from abc import ABC, abstractmethod


class VectorDBInterface(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def count(self, collection: str) -> int: ...

    @abstractmethod
    async def existing_ids(self, collection: str, ids: list[str]) -> set[str]: ...

    @abstractmethod
    async def insert(self, collection: str, ids: list[str], texts: list[str],
                     vectors: list[list[float]], metadatas: list[dict]) -> None: ...

    @abstractmethod
    async def search(self, collection: str, vector: list[float], limit: int,
                     where: dict | None = None) -> list[dict]: ...

    @abstractmethod
    async def delete_collection(self, collection: str) -> None: ...

    @abstractmethod
    async def list_collections(self) -> dict[str, int]: ...
