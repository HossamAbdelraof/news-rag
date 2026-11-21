import json

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.routes.schemes import ChatRequest

chat_router = APIRouter(prefix="/ai/v1", tags=["chat"])

SAVED_FIELDS = ("question", "status", "reason", "search_question", "sources", "timings",
                "model", "reranked", "reranker_device")


def unavailable(message: str) -> JSONResponse:
    return JSONResponse(content={"signal": message}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


async def open_thread(clients, thread_id: str | None) -> tuple[dict, list]:
    thread = await clients.history.get_thread(thread_id) if thread_id else None
    if thread is None:
        thread = await clients.history.create_thread()
    history = await clients.history.get_history(thread["id"])
    return thread, history


async def save_turn(clients, thread_id: str, question: str, result: dict) -> None:
    details = {key: result[key] for key in SAVED_FIELDS}
    await clients.history.add_turn(thread_id, question, result["answer"], details)


def get_chat(clients):
    if clients.history is None:
        return None, unavailable("Chat threads need Postgres (is Docker running?)")
    try:
        return clients.chat_controller(), None
    except ValueError as e:  # CHAT_API_KEY missing
        return None, unavailable(str(e))


@chat_router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> JSONResponse:
    clients = request.app.state.clients
    controller, error = get_chat(clients)
    if error:
        return error

    thread, history = await open_thread(clients, body.thread_id)
    result = await controller.answer(body.question, history, callbacks=clients.monitoring.callbacks(),
                                     session_id=thread["id"], rerank=body.rerank)
    await save_turn(clients, thread["id"], body.question, result)
    return JSONResponse(content={"signal": result["status"], "thread_id": thread["id"], **result})


@chat_router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    """Same as /chat, but sends one JSON event per line while the answer is written."""
    clients = request.app.state.clients
    controller, error = get_chat(clients)
    if error:
        return error

    thread, history = await open_thread(clients, body.thread_id)

    async def events():
        yield json.dumps({"type": "thread", "thread_id": thread["id"]}) + "\n"
        async for event in controller.stream(body.question, history, callbacks=clients.monitoring.callbacks(),
                                             session_id=thread["id"], rerank=body.rerank):
            if event["type"] == "final":
                await save_turn(clients, thread["id"], body.question, event)
                event = {**event, "thread_id": thread["id"]}
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@chat_router.get("/threads")
async def list_threads(request: Request) -> JSONResponse:
    history = request.app.state.clients.history
    if history is None:
        return unavailable("Chat threads need Postgres")
    return JSONResponse(content={"threads": await history.list_threads()})


@chat_router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str) -> JSONResponse:
    history = request.app.state.clients.history
    if history is None:
        return unavailable("Chat threads need Postgres")
    thread = await history.get_thread(thread_id)
    if thread is None:
        return JSONResponse(content={"signal": "Thread not found"}, status_code=status.HTTP_404_NOT_FOUND)
    return JSONResponse(content={"thread": thread, "messages": await history.get_messages(thread_id)})


@chat_router.delete("/threads/{thread_id}")
async def delete_thread(request: Request, thread_id: str) -> JSONResponse:
    history = request.app.state.clients.history
    if history is None:
        return unavailable("Chat threads need Postgres")
    if await history.delete_thread(thread_id):
        return JSONResponse(content={"signal": "Thread deleted"})
    return JSONResponse(content={"signal": "Thread not found"}, status_code=status.HTTP_404_NOT_FOUND)
