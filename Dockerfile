# API + chat page. The reranker is downloaded at build time, so the container runs offline.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# dependencies first, so they are cached
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra local

# reranker model
RUN uv run --no-sync python -c "from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-reranker-v2-m3', allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer*', 'sentencepiece*'])"

COPY src ./src
COPY scripts ./scripts
COPY ui ./ui

ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
