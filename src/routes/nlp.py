from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.controllers import ProcessController
from src.helpers.config import get_settings
from src.routes.schemes import IndexRequest, SearchRequest

nlp_router = APIRouter(prefix="/ai/v1/nlp", tags=["nlp"])


@nlp_router.post("/index")
async def index_files(request: Request, body: IndexRequest) -> JSONResponse:
    # Step 1 - read the files and cut them into chunks
    ok, signal, status_code, chunks, stats = ProcessController().process_path(body.path)
    if not ok:
        return JSONResponse(content={"signal": signal, "stats": stats}, status_code=status_code)

    # Step 2 - embed and store the chunks
    ok, signal, status_code, info = await request.app.state.clients.nlp_controller().index_chunks(chunks, body.reset)
    return JSONResponse(content={"signal": signal, "files": stats, "index": info}, status_code=status_code)


@nlp_router.post("/search")
async def search(request: Request, body: SearchRequest) -> JSONResponse:
    defaults = get_settings().search
    ok, signal, status_code, result = await request.app.state.clients.nlp_controller().search(
        body.query, candidates=body.candidates or defaults.SEARCH_CANDIDATES,
        top_k=body.top_k or defaults.SEARCH_TOP_K, rerank=body.rerank, category=body.category,
    )
    if not ok:
        return JSONResponse(content={"signal": signal}, status_code=status_code)
    for key in ("before_rerank", "after_rerank"):
        result[key] = [hit.model_dump() for hit in result[key]]
    return JSONResponse(content={"signal": signal, **result}, status_code=status_code)
