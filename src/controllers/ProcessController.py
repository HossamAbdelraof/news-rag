import hashlib
import io
import logging
import re
from pathlib import Path

from fastapi import status

from src.models import Chunk, ResponseSignal

from .BaseController import BaseController

logger = logging.getLogger("uvicorn.error")

SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".pptx", ".html", ".htm"}
R8_LINE = re.compile(r"^[a-z][a-z-]*\t\S")


class ProcessController(BaseController):
    """Turns files into chunks. R8 files have one article per line ("label<TAB>text"),
    every other file is read with Docling."""

    def __init__(self):
        super().__init__()
        self.config = self.settings.chunk
        self.chunker = None

    # ---- reading files

    @staticmethod
    def collect_files(path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in SUPPORTED else []
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)

    @staticmethod
    def is_r8_file(path: Path) -> bool:
        if path.suffix.lower() != ".txt":
            return False
        with path.open(encoding="utf-8", errors="ignore") as f:
            lines = [line for _, line in zip(range(20), f) if line.strip()]
        return bool(lines) and all(R8_LINE.match(line) for line in lines)

    @staticmethod
    def clean_r8_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"\s*\breuter$", "", text).strip()

    def read_r8(self, path: Path) -> list[dict]:
        split = "test" if "test" in path.stem else "train" if "train" in path.stem else path.stem
        records = []
        for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
            label, _, text = line.rstrip("\n").partition("\t")
            text = self.clean_r8_text(text)
            if text:
                records.append({"doc_id": f"r8-{split}-{line_no:05d}", "category": label.strip(), "text": text})
        return records

    # ---- chunking

    @property
    def body_budget(self) -> int:
        # Docling cuts each chunk to this size, the overlap is added afterwards.
        return self.config.CHUNK_MAX_TOKENS - self.config.CHUNK_OVERLAP_TOKENS

    def get_chunker(self):
        if self.chunker is None:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()  # the tokenizer only counts tokens, its length warning is noise
            tokenizer = HuggingFaceTokenizer.from_pretrained(self.config.CHUNK_TOKENIZER_ID, max_tokens=self.body_budget)
            self.chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        return self.chunker

    def count_tokens(self, text: str) -> int:
        return self.get_chunker().tokenizer.count_tokens(text)

    def merge_small_tail(self, bodies: list[str], budget: int) -> list[str]:
        """Join a tiny last piece to the one before it, if they fit together."""
        if len(bodies) < 2 or self.count_tokens(bodies[-1]) >= self.config.CHUNK_MIN_TAIL_TOKENS:
            return bodies
        joined = f"{bodies[-2]} {bodies[-1]}"
        if self.count_tokens(joined) <= budget:
            return bodies[:-2] + [joined]
        return bodies

    def overlap_tail(self, text: str) -> str:
        """The last words of `text` that fit in CHUNK_OVERLAP_TOKENS."""
        if self.config.CHUNK_OVERLAP_TOKENS <= 0:
            return ""
        tail, size = [], 0
        for word in reversed(text.split()):
            size += self.count_tokens(word)
            if size > self.config.CHUNK_OVERLAP_TOKENS:
                break
            tail.insert(0, word)
        return " ".join(tail)

    def chunk_document(self, document, doc_id: str, source_file: str, category: str = "",
                       with_headings: bool = True) -> list[Chunk]:
        parts = []
        for c in self.get_chunker().chunk(document):
            body = re.sub(r"\s+", " ", c.text).strip()
            heading = " ".join(c.meta.headings or []) if with_headings else ""
            if body:
                parts.append((body, heading))
        if not parts:
            return []

        bodies = [body for body, _ in parts]
        headings = [heading for _, heading in parts]
        if len(set(headings)) == 1:  # one section only, so pieces can be merged safely
            budget = self.body_budget - (self.count_tokens(headings[0]) if headings[0] else 0)
            bodies = self.merge_small_tail(bodies, budget)
            headings = headings[:len(bodies)]

        # final text = heading + end of the previous chunk + this chunk
        chunks = []
        for i, (body, heading) in enumerate(zip(bodies, headings)):
            overlap = self.overlap_tail(bodies[i - 1]) if i > 0 else ""
            text = " ".join(p for p in (heading, overlap, body) if p)
            chunks.append(Chunk(doc_id=doc_id, chunk_order=i + 1, text=text, source_file=source_file,
                                category=category))
        return chunks

    @staticmethod
    def convert(sources: list) -> list:
        from docling.document_converter import DocumentConverter

        return [r.document for r in DocumentConverter().convert_all(sources)]

    # ---- main entry

    def process_r8_file(self, file: Path, seen_texts: set[str], stats: dict) -> list[Chunk]:
        from docling_core.types.io import DocumentStream

        records = []
        for r in self.read_r8(file):
            text_hash = hashlib.sha1(r["text"].encode()).hexdigest()
            if text_hash in seen_texts:
                stats["r8_duplicates_skipped"] += 1
                continue
            seen_texts.add(text_hash)
            records.append(r)

        streams = [DocumentStream(name=f"{r['doc_id']}.md", stream=io.BytesIO(f"# {r['category']}\n\n{r['text']}\n".encode()))
                   for r in records]
        chunks = []
        for r, doc in zip(records, self.convert(streams), strict=True):
            # the category is kept in metadata, not in the chunk text
            chunks += self.chunk_document(doc, r["doc_id"], file.name, r["category"], with_headings=False)

        stats["r8_articles"] += len(records)
        logger.info("%s: %d R8 articles", file.name, len(records))
        return chunks

    def process_document(self, file: Path, stats: dict) -> list[Chunk]:
        from docling_core.types.io import DocumentStream

        source = file
        if file.suffix.lower() == ".txt":  # Docling reads plain text as Markdown
            source = DocumentStream(name=f"{file.stem}.md", stream=io.BytesIO(file.read_bytes()))
        doc = self.convert([source])[0]
        stats["documents"] += 1
        logger.info("%s: document", file.name)
        return self.chunk_document(doc, file.name, file.name)

    def process_path(self, path: str) -> tuple[bool, str, int, list[Chunk], dict]:
        root = Path(path)
        if not root.exists():
            return False, ResponseSignal.path_not_found.value, status.HTTP_404_NOT_FOUND, [], {}
        files = self.collect_files(root)
        if not files:
            return False, ResponseSignal.no_files_found.value, status.HTTP_404_NOT_FOUND, [], {}

        chunks = []
        stats = {"files": len(files), "r8_articles": 0, "r8_duplicates_skipped": 0, "documents": 0}
        seen_texts = set()
        try:
            for file in files:
                if self.is_r8_file(file):
                    chunks += self.process_r8_file(file, seen_texts, stats)
                else:
                    chunks += self.process_document(file, stats)
        except Exception as e:
            logger.exception("Processing %s failed: %s", path, e)
            return False, ResponseSignal.processing_failed.value, status.HTTP_500_INTERNAL_SERVER_ERROR, [], stats

        stats["chunks"] = len(chunks)
        return True, ResponseSignal.index_success.value, status.HTTP_200_OK, chunks, stats
