import re

from src.helpers.config import VectorDBConfig

from .providers import ChromaDBProvider
from .VectorDBInterface import VectorDBInterface


class VectorDBProviderFactory:
    def __init__(self, config: VectorDBConfig, db_path: str):
        self.config = config
        self.db_path = db_path

    def create(self) -> VectorDBInterface:
        return ChromaDBProvider(path=self.db_path)

    def collection_name(self, embedding_model_id: str) -> str:
        """One collection per embedding model, so two models never mix vectors."""
        slug = re.sub(r"[^a-z0-9.-]+", "-", embedding_model_id.lower().removeprefix("text-embedding-")).strip("-.")
        return f"{self.config.VECTOR_DB_COLLECTION_PREFIX}_{slug}"
