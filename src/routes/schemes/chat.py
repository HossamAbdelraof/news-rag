from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=40, description="omit to start a new thread")
    rerank: bool = Field(default=True, description="use the bge reranker (GPU if available, else CPU)")
