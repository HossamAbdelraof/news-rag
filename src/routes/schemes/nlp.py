from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    path: str = Field(description="File or folder on the server, e.g. ../datasets")
    reset: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    candidates: int | None = Field(default=None, ge=1, le=200, description="chunks sent to the reranker")
    top_k: int | None = Field(default=None, ge=1, le=50, description="results returned")
    rerank: bool = True
    category: str | None = Field(default=None, description="optional R8 category filter, e.g. crude")
