# Financial News Assistant (RAG)

Ask questions about financial news and get short answers with their sources.
The answers come only from the Reuters R8 news dataset (1987). If the answer is not in the
news, the assistant says so instead of guessing.

It works in English and Arabic.

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

## How it works

### 1. Indexing

```
 file / folder
      │
      v
┌────────────────┐   R8 .txt: one article per line ("label<TAB>text")
│  read files    │   other files (.pdf .docx .pptx .md .html): Docling
└───────┬────────┘
        v
┌────────────────┐   300 tokens per chunk, 50 tokens overlap
│  chunk         │   duplicate articles are indexed once
└───────┬────────┘
        v
┌────────────────┐   Qwen3-Embedding-4B
│  embed         │
└───────┬────────┘
        v
┌────────────────┐   one collection per embedding model
│  store (Chroma)│   chunks already stored are skipped
└────────────────┘
```

### 2. Chat

Every question goes through the same fixed steps:

```
                 question
                    │
        ┌───────────┴────────────┐
        v                        v
┌───────────────┐        ┌───────────────┐
│ input guard   │        │ rewrite       │  follow-up -> full question
│ advice? off-  │        └───────┬───────┘
│ topic?        │                v
└───────┬───────┘        ┌───────────────┐
        │                │ search        │  30 chunks -> rerank -> top 5
        │                └───────┬───────┘
        └───────────┬────────────┘
                    v
            ┌───────────────┐
            │ relevance     │──── low score ───> "not found in the sources"
            │ gate          │──── blocked ─────> polite refusal
            └───────┬───────┘
                    v
            ┌───────────────┐
            │ generate      │  Gemma 4, answers only from the sources
            └───────┬───────┘
                    v
            ┌───────────────┐
            │ output guard  │  real citations? no made-up numbers?
            └───────┬───────┘  (fails -> write again once)
                    v
              answer + sources
```

- **Input guard:** NeMo Guardrails compares the question with example phrases. It does not call
  the LLM.
- **Relevance gate:** if the best reranker score is below `0.1`, the LLM is not called at all.
- **Output guard:** every `[n]` must point to a real source. The news has no figures, so any
  number in the answer counts as made up.

## Models

| Part | Model | Why |
|---|---|---|
| Embedding | `qwen/qwen3-embedding-4b` | Best in our benchmark, and it handles Arabic |
| Reranker | `BAAI/bge-reranker-v2-m3` | Best in our benchmark, runs on the GPU or the CPU |
| Chat | `google/gemma-4-26b-a4b-it` | Fast and cheap on OpenRouter |

All models are set in `.env`. To try another model, change the name there.

## Results

**Retrieval**, on 96 test questions (nDCG@10):

| Method | Score |
|---|---|
| Keyword search (BM25) | 0.817 |
| Vector search | 0.911 |
| Vector search + reranker | **0.951** |

**Chat**, 40 normal questions and 24 that should not be answered:

| Check | Result |
|---|---|
| Normal questions answered | 39 / 40 |
| Answers that cite the right article | 38 / 39 |
| Investment advice refused | 6 / 6 |
| Off-topic questions refused | 6 / 6 |

You can run the checks yourself with `scripts/evaluate.py` and `scripts/evaluate_chat.py`.

## Project Structure

```
news-rag/
├── datasets/               R8 train and test files
├── data/eval/              test questions
├── scripts/
│   ├── index_files.py      index a file or folder
│   ├── ask.py              search, before and after reranking
│   ├── chat.py             chat in the terminal
│   ├── evaluate.py         retrieval scores
│   └── evaluate_chat.py    chat and guard scores
├── src/
│   ├── main.py             FastAPI app
│   ├── controllers/        chunking, search, chat flow
│   ├── models/             data classes and chat history
│   ├── routes/             API endpoints
│   ├── stores/             embedding, Chroma, reranker, guards, prompts
│   └── helpers/config.py   settings from .env
├── ui/                     web page
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Prerequisites

- An [OpenRouter](https://openrouter.ai) API key (for embeddings and chat)
- Docker, **or** Python 3.12 with [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU is optional. The reranker also runs on the CPU, but more slowly.

## Run with Docker (recommended)

```bash
cp .env.example .env          # then put your key in CHAT_API_KEY and set POSTGRES_PASSWORD
docker compose up -d --build
docker compose exec api uv run --no-sync python -m scripts.index_files /data
```

Open http://localhost:8000.

With an NVIDIA GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Run locally

```bash
uv sync
cp .env.example .env          # put your key in CHAT_API_KEY

uv run python -m scripts.index_files datasets
uv run python -m scripts.ask "Why did OPEC cut oil output?"
uv run python -m scripts.chat
uv run uvicorn src.main:app --reload
```

The chat page is at http://localhost:8000 and the API docs are at http://localhost:8000/docs.
Saved chat threads need Postgres. Search and `scripts.chat` work without it.

## API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/ai/v1/` | Health check and active models |
| `POST` | `/ai/v1/nlp/index` | Index a file or folder |
| `POST` | `/ai/v1/nlp/search` | Search, with results before and after reranking |
| `POST` | `/ai/v1/chat` | Ask a question |
| `POST` | `/ai/v1/chat/stream` | Same, but streams the answer |
| `GET` / `DELETE` | `/ai/v1/threads/{id}` | Read or delete a chat thread |

Example:

```bash
curl -X POST http://localhost:8000/ai/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Why did OPEC cut oil output?"}'
```

## Tests

```bash
uv run pytest
```

## Limitations

- The news is from 1987, so it can't answer questions about today.
- The input guard catches most advice questions, but few off-topic ones. The relevance gate
  stops most of the rest.
- With OpenRouter, questions and news text leave the machine. A bank would run the same models
  on its own servers (for example with vLLM). Only `.env` would change.
- The chunk size was tested only on short news articles, not on long reports.
