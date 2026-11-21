from enum import Enum


class ResponseSignal(Enum):
    health_ok = "Service is healthy"

    # processing / indexing
    path_not_found = "File or folder not found"
    no_files_found = "No supported files found"
    processing_failed = "Failed to read or chunk the files"
    index_success = "Files indexed successfully"
    index_failed = "Indexing failed"
    embedding_failed = "Embedding request failed (check the embedding API and key)"

    # search
    search_success = "Search completed"
    search_failed = "Search failed"
    collection_empty = "Nothing indexed for this embedding model yet; run indexing first"
    rerank_failed = "Reranking failed"
