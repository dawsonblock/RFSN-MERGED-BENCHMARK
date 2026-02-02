"""Retrieval module - operational memory for planning."""
from .embeddings import hash_embed, cosine, batch_cosine
from .failure_index import FailureIndex, FailureRecord
from .recall import build_retrieval_context, format_retrieval_for_prompt, extract_retrieval_insights

__all__ = [
    "hash_embed",
    "cosine",
    "batch_cosine",
    "FailureIndex",
    "FailureRecord",
    "build_retrieval_context",
    "format_retrieval_for_prompt",
    "extract_retrieval_insights",
]
