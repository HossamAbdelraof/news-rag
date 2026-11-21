from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from src.models import ResponseSignal

base_router = APIRouter(prefix="/ai/v1", tags=["base"])


@base_router.get("/")
async def health(request: Request) -> JSONResponse:
    clients = request.app.state.clients
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "signal": ResponseSignal.health_ok.value,
        "embedding_model": clients.embedding.model_id,
        "collection": clients.collection,
        "reranker_model": clients.reranker.model_id if clients.reranker else None,
        "collections": await clients.vectordb.list_collections(),
    })
