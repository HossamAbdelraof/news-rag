# Financial News Assistant (RAG)

Ask questions about financial news and get short answers with their sources.
The answers come from the Reuters R8 news dataset. It works in English and Arabic.

## System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  R8 dataset  │────>│   Docling    │────>│    Qwen3     │────>│    Chroma    │
│  PDF / DOCX  │     │   chunking   │     │  embeddings  │     │   vector DB  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      v
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web UI /   │<───>│   FastAPI    │<───>│  LangGraph   │<───>│ bge reranker │
│   terminal   │     │              │     │  chat flow   │     │              │
└──────────────┘     └──────┬───────┘     └──────┬───────┘     └──────────────┘
                            │                    │
                            v                    v
                     ┌──────────────┐     ┌──────────────┐
                     │   Postgres   │     │   Gemma 4    │
                     │ chat threads │     │ (OpenRouter) │
                     └──────────────┘     └──────────────┘
```

## Chat Flow

```
              question
                 │
       ┌─────────┴─────────┐
       v                   v
 ┌───────────┐       ┌───────────┐
 │input guard│       │  rewrite  │
 └─────┬─────┘       └─────┬─────┘
       │                   v
       │             ┌───────────┐
       │             │  search   │
       │             └─────┬─────┘
       └─────────┬─────────┘
                 v
          ┌─────────────┐
          │relevance    │───> refuse / "not found"
          │gate         │
          └──────┬──────┘
                 v
          ┌─────────────┐
          │  generate   │
          └──────┬──────┘
                 v
          ┌─────────────┐
          │output guard │
          └──────┬──────┘
                 v
          answer + sources
```

## Models

| Part | Model |
|---|---|
| Embedding | `qwen/qwen3-embedding-4b` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Chat | `google/gemma-4-26b-a4b-it` |

The models are set in `.env`.

## Project Structure

```
news-rag/
├── datasets/          R8 train and test files
├── data/eval/         test questions
├── scripts/           index, ask, chat and evaluate from the terminal
├── src/
│   ├── main.py        FastAPI app
│   ├── controllers/   chunking, search, chat flow
│   ├── models/        data classes and chat history
│   ├── routes/        API endpoints
│   ├── stores/        embedding, Chroma, reranker, guards, prompts
│   └── helpers/       settings
├── ui/                web page
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai) API key
- PostgreSQL, for saving the chat threads

## How to run

**1. Install**

```bash
uv sync
```

**2. Configure**

```bash
cp .env.example .env
```

In `.env`, add your OpenRouter key to `CHAT_API_KEY` and your Postgres details to `POSTGRES_*`.

**3. Index the news** (once)

```bash
uv run python -m scripts.index_files datasets
```

If it stops halfway, run it again. It continues where it stopped.

**4. Start the app**

```bash
uv run uvicorn src.main:app
```

Open http://localhost:8000 and ask a question. The first question takes about a minute,
because the reranker loads. After that, answers take a few seconds.

**Or use the terminal** (no Postgres needed):

```bash
uv run python -m scripts.chat
uv run python -m scripts.ask "Why did OPEC cut oil output?"
```

## API

| Method | Path | |
|---|---|---|
| `POST` | `/ai/v1/nlp/index` | Index a file or folder |
| `POST` | `/ai/v1/nlp/search` | Search |
| `POST` | `/ai/v1/chat` | Ask a question |
| `GET` | `/ai/v1/threads` | List chat threads |

API docs: http://localhost:8000/docs

## Tests

```bash
uv run pytest
```
