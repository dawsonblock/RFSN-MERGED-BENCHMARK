"""Lightweight embeddings for failure similarity matching.

Enhanced with bigrams and better normalization.
"""
from __future__ import annotations
import re
import math


def _tokens(text: str) -> list[str]:
    """Extract tokens from text for hashing."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    toks = [t for t in text.split() if t and len(t) >= 2]
    return toks[:3000]


def _bigrams(tokens: list[str]) -> list[str]:
    """Generate simple bigrams from tokens."""
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]


def hash_embed(text: str, dim: int = 4096) -> list[float]:
    """
    Deterministic hashed bag-of-words/bigrams with log scaling.
    
    Uses 4096 dimension for reduced collisions.
    Includes bigrams for rudimentary phase matching.
    
    Args:
        text: Input text to embed
        dim: Embedding dimension (default 4096)
        
    Returns:
        L2-normalized embedding vector
    """
    v = [0.0] * dim
    
    toks = _tokens(text)
    grams = _bigrams(toks)
    
    # Embed unigrams (weight 1.0)
    for tok in toks:
        h = hash(tok) % dim
        v[h] += 1.0
        
    # Embed bigrams (weight 0.5 - subtle structure boost)
    for gram in grams:
        h = hash(gram) % dim
        v[h] += 0.5
    
    # Log scaling + L2 normalize
    norm = 0.0
    for i, x in enumerate(v):
        if x > 0:
            # TF-like log scaling: 1 + log(count)
            x_val = 1.0 + math.log(x)
            v[i] = x_val
            norm += x_val * x_val
    
    norm = math.sqrt(norm) if norm > 0 else 1.0
    return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # Assuming standard dot product for L2 normalized vectors
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False))


def batch_cosine(query: list[float], vectors: list[list[float]]) -> list[float]:
    """Compute cosine similarity between query and multiple vectors."""
    return [cosine(query, v) for v in vectors]
