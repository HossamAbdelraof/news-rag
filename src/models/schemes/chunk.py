import hashlib

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    doc_id: str  # "r8-test-00042" or "report.pdf"
    chunk_order: int
    text: str
    source_file: str
    category: str = ""

    @property
    def id(self) -> str:
        """Same chunk -> same id, so re-indexing skips it."""
        return hashlib.sha1(f"{self.doc_id}#{self.chunk_order}".encode()).hexdigest()

    def metadata(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_order": self.chunk_order,
                "source_file": self.source_file, "category": self.category}


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    vector_rank: int
    vector_score: float
    rerank_score: float | None = None
