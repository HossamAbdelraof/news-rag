import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, delete, func, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ThreadTable(Base):
    __tablename__ = "chat_threads"
    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    title: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MessageTable(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _thread(row: ThreadTable, questions: int | None = None) -> dict:
    out = {"id": row.id, "title": row.title, "created_at": row.created_at.isoformat(timespec="seconds"),
           "updated_at": row.updated_at.isoformat(timespec="seconds")}
    if questions is not None:
        out["questions"] = questions
    return out


class ChatHistoryModel:
    def __init__(self, sessions):
        self.sessions = sessions

    async def create_tables(self, engine) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # ---- threads
    async def create_thread(self, title: str = "New chat") -> dict:
        row = ThreadTable(id=uuid.uuid4().hex[:12], title=title, created_at=_now(), updated_at=_now())
        async with self.sessions() as s, s.begin():
            s.add(row)
        return _thread(row)

    async def get_thread(self, thread_id: str) -> dict | None:
        async with self.sessions() as s:
            row = await s.get(ThreadTable, thread_id)
        return _thread(row) if row else None

    async def list_threads(self, limit: int = 50) -> list[dict]:
        questions = func.count(MessageTable.id) / 2
        stmt = (select(ThreadTable, questions).outerjoin(MessageTable)
                .group_by(ThreadTable.id).order_by(ThreadTable.updated_at.desc()).limit(limit))
        async with self.sessions() as s:
            rows = (await s.execute(stmt)).all()
        return [_thread(t, int(q)) for t, q in rows]

    async def delete_thread(self, thread_id: str) -> bool:
        async with self.sessions() as s, s.begin():
            result = await s.execute(delete(ThreadTable).where(ThreadTable.id == thread_id))
        return result.rowcount > 0

    # ---- messages
    async def add_turn(self, thread_id: str, question: str, answer: str, details: dict) -> None:
        """The first question becomes the thread title."""
        now = _now()
        async with self.sessions() as s, s.begin():
            count = await s.scalar(select(func.count()).where(MessageTable.thread_id == thread_id))
            values = {"updated_at": now} if count else {"updated_at": now, "title": question[:80]}
            await s.execute(update(ThreadTable).where(ThreadTable.id == thread_id).values(**values))
            s.add_all([
                MessageTable(thread_id=thread_id, role="user", content=question, details={}, created_at=now),
                MessageTable(thread_id=thread_id, role="assistant", content=answer, details=details, created_at=now),
            ])

    async def get_messages(self, thread_id: str) -> list[dict]:
        stmt = select(MessageTable).where(MessageTable.thread_id == thread_id).order_by(MessageTable.id)
        async with self.sessions() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [{"role": r.role, "content": r.content, "details": r.details,
                 "created_at": r.created_at.isoformat(timespec="seconds")} for r in rows]

    async def get_history(self, thread_id: str) -> list[tuple[str, str]]:
        """Answered (question, answer) pairs, used for follow-up questions."""
        messages = await self.get_messages(thread_id)
        return [(u["content"], a["content"]) for u, a in zip(messages[::2], messages[1::2])
                if a["details"].get("status") == "answered"]


class InMemoryChatHistory:
    """Same methods as ChatHistoryModel, kept in memory. Used when Postgres is not reachable
    (e.g. on Vercel without a database): threads work but are lost when the instance restarts."""

    def __init__(self):
        self.threads: dict[str, dict] = {}
        self.messages: dict[str, list[dict]] = {}

    async def create_thread(self, title: str = "New chat") -> dict:
        now = _now().isoformat(timespec="seconds")
        thread = {"id": uuid.uuid4().hex[:12], "title": title, "created_at": now, "updated_at": now}
        self.threads[thread["id"]], self.messages[thread["id"]] = thread, []
        return dict(thread)

    async def get_thread(self, thread_id: str) -> dict | None:
        thread = self.threads.get(thread_id)
        return dict(thread) if thread else None

    async def list_threads(self, limit: int = 50) -> list[dict]:
        rows = sorted(self.threads.values(), key=lambda t: t["updated_at"], reverse=True)[:limit]
        return [{**t, "questions": len(self.messages[t["id"]]) // 2} for t in rows]

    async def delete_thread(self, thread_id: str) -> bool:
        self.messages.pop(thread_id, None)
        return self.threads.pop(thread_id, None) is not None

    async def add_turn(self, thread_id: str, question: str, answer: str, details: dict) -> None:
        thread = self.threads.get(thread_id)
        if thread is None:
            return
        now = _now().isoformat(timespec="seconds")
        if not self.messages[thread_id]:
            thread["title"] = question[:80]
        thread["updated_at"] = now
        self.messages[thread_id] += [{"role": "user", "content": question, "details": {}, "created_at": now},
                                     {"role": "assistant", "content": answer, "details": details, "created_at": now}]

    async def get_messages(self, thread_id: str) -> list[dict]:
        return [dict(m) for m in self.messages.get(thread_id, [])]

    get_history = ChatHistoryModel.get_history
