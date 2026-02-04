"""Advanced embedding layer with CodeBERT and FAISS support.

Provides high-quality code embeddings using CodeBERT and optional
FAISS indexing for fast similarity search at scale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional dependencies - graceful fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    torch = None  # type: ignore


# Log dependency status at module load
def _log_embedding_dependencies() -> None:
    """Log embedding dependency status with install suggestions."""
    # Check for skip flag
    skip_codebert = os.environ.get("RFSN_SKIP_CODEBERT", "").lower() in ("1", "true", "yes")
    if skip_codebert:
        logger.info("RFSN_SKIP_CODEBERT=1: Using hash embeddings (fast mode)")
        return
    
    missing = []
    
    if not HAS_NUMPY:
        missing.append("numpy")
    if not HAS_FAISS:
        missing.append("faiss-cpu (IVF indexing)")
    if not HAS_TRANSFORMERS:
        missing.append("transformers torch (CodeBERT)")
    
    if missing:
        logger.warning(
            f"Embedding dependencies missing: {', '.join(missing)}\n"
            "Using fallback hash embeddings (lower quality).\n"
            "Install for semantic search:\n"
            "  pip install numpy transformers torch faiss-cpu"
        )
    else:
        logger.info("All embedding dependencies available (numpy, transformers, torch, faiss)")


_log_embedding_dependencies()


@dataclass
class EmbeddingConfig:
    """Configuration for the advanced embedding layer."""
    model_name: str = "microsoft/codebert-base"
    cache_dir: str = ".rfsn_state/embeddings_cache"
    embedding_dim: int = 768  # CodeBERT dimension
    max_length: int = 512
    use_faiss: bool = True
    faiss_nlist: int = 100  # Number of IVF clusters
    device: str = "cpu"  # or "cuda" if available


class CodeBERTEmbedder:
    """
    CodeBERT-based embedding generator for code and error signatures.
    
    Provides high-quality semantic embeddings that understand code structure
    and programming concepts, significantly improving similarity matching
    over bag-of-words approaches.
    
    Falls back to hash-based embeddings if transformers not available.
    """
    
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        self._cache_path = Path(self.config.cache_dir) / "embedding_cache.json"
        
        # Load cache from disk if exists
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        try:
            if self._cache_path.exists():
                with open(self._cache_path) as f:
                    self._cache = json.load(f)
                logger.debug("Loaded %d cached embeddings", len(self._cache))
        except Exception as e:
            logger.warning("Failed to load embedding cache: %s", e)
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk (periodic)."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Only save last 10k entries to prevent unbounded growth
            cache_to_save = dict(list(self._cache.items())[-10000:])
            with open(self._cache_path, "w") as f:
                json.dump(cache_to_save, f)
        except Exception as e:
            logger.warning("Failed to save embedding cache: %s", e)
    
    def _ensure_model(self) -> bool:
        """Lazy load the model."""
        if self._model is not None:
            return True
        
        # Check skip flag
        skip_codebert = os.environ.get("RFSN_SKIP_CODEBERT", "").lower() in ("1", "true", "yes")
        if skip_codebert:
            return False
        
        if not HAS_TRANSFORMERS:
            logger.info("Transformers not available, using hash embeddings")
            return False
        
        with self._lock:
            if self._model is not None:
                return True
            
            try:
                logger.info("Loading CodeBERT model: %s", self.config.model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self._model = AutoModel.from_pretrained(self.config.model_name)
                self._model.eval()
                
                # Move to device
                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                    logger.info("CodeBERT loaded on CUDA")
                else:
                    logger.info("CodeBERT loaded on CPU")
                
                return True
            except Exception as e:
                logger.warning("Failed to load CodeBERT: %s, falling back to hash", e)
                return False
    
    def _cache_key(self, text: str) -> str:
        """Create a cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def _hash_embed(self, text: str, dim: int = 768) -> list[float]:
        """Fallback hash-based embedding."""
        import math
        import re
        
        # Tokenize
        text_lower = (text or "").lower()
        text_clean = re.sub(r"[^a-z0-9_]+", " ", text_lower)
        tokens = [t for t in text_clean.split() if t and len(t) >= 2][:500]
        
        # Hash to vector
        vec = [0.0] * dim
        for tok in tokens:
            h = hash(tok) % dim
            vec[h] += 1.0
        
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.
        
        Uses CodeBERT if available, falls back to hash embedding.
        Results are cached for efficiency.
        
        Args:
            text: Input text (code, error message, etc.)
            
        Returns:
            Embedding vector (768-dim for CodeBERT, configurable for hash)
        """
        if not text or not text.strip():
            return [0.0] * self.config.embedding_dim
        
        # Check cache
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        if self._ensure_model():
            embedding = self._embed_with_codebert(text)
        else:
            embedding = self._hash_embed(text, self.config.embedding_dim)
        
        # Cache result
        self._cache[cache_key] = embedding
        
        # Periodic cache save (every 100 new entries)
        if len(self._cache) % 100 == 0:
            self._save_cache()
        
        return embedding
    
    def _embed_with_codebert(self, text: str) -> list[float]:
        """Generate embedding using CodeBERT."""
        # Truncate text to max length
        text = text[:self.config.max_length * 4]  # Rough char estimate
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
        )
        
        # Move to device
        if self.config.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            
            # L2 normalize
            embedding = embedding / (embedding.norm() + 1e-8)
            
            return embedding.cpu().tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        More efficient than calling embed() multiple times
        when CodeBERT is available.
        """
        if not texts:
            return []
        
        # Check cache for all
        results = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Generate embeddings for uncached
        if uncached_texts:
            if self._ensure_model():
                new_embeddings = self._batch_embed_codebert(uncached_texts)
            else:
                new_embeddings = [
                    self._hash_embed(t, self.config.embedding_dim)
                    for t in uncached_texts
                ]
            
            # Cache and add to results
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings, strict=False):
                cache_key = self._cache_key(text)
                self._cache[cache_key] = emb
                results.append((idx, emb))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]
    
    def _batch_embed_codebert(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """Batch embedding with CodeBERT."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate texts
            batch = [t[:self.config.max_length * 4] for t in batch]
            
            # Tokenize
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True,
            )
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # L2 normalize
                norms = embeddings.norm(dim=1, keepdim=True) + 1e-8
                embeddings = embeddings / norms
                
                all_embeddings.extend(embeddings.cpu().tolist())
        
        return all_embeddings


class FAISSIndex:
    """
    FAISS-based vector index for fast similarity search.
    
    Uses IVF (Inverted File) index for O(log n) approximate
    nearest neighbor search, scaling to millions of vectors.
    
    Falls back to brute-force numpy search if FAISS unavailable.
    """
    
    def __init__(
        self,
        dim: int = 768,
        nlist: int = 100,
        use_gpu: bool = False,
    ):
        self.dim = dim
        self.nlist = nlist
        self.use_gpu = use_gpu
        self._index = None
        self._trained = False
        self._vectors: list[list[float]] = []
        self._metadata: list[dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def _ensure_index(self) -> None:
        """Initialize the FAISS index."""
        if self._index is not None:
            return
        
        with self._lock:
            if self._index is not None:
                return
            
            if HAS_FAISS:
                # Create IVF index
                quantizer = faiss.IndexFlatL2(self.dim)
                self._index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
                logger.info("Created FAISS IVF index (dim=%d, nlist=%d)", self.dim, self.nlist)
            else:
                # Fallback marker
                self._index = "numpy_fallback"
                logger.info("FAISS not available, using numpy brute-force search")
    
    def add(self, vector: list[float], metadata: dict[str, Any] | None = None) -> int:
        """
        Add a vector to the index.
        
        Args:
            vector: Embedding vector
            metadata: Optional metadata to associate
            
        Returns:
            Index of the added vector
        """
        self._ensure_index()
        
        with self._lock:
            idx = len(self._vectors)
            self._vectors.append(vector)
            self._metadata.append(metadata or {})
            
            # For FAISS, we need to train then add
            if HAS_FAISS and self._index != "numpy_fallback":
                if not self._trained and len(self._vectors) >= self.nlist:
                    self._train_index()
                
                if self._trained:
                    vec_np = np.array([vector], dtype=np.float32)
                    self._index.add(vec_np)
            
            return idx
    
    def add_batch(
        self,
        vectors: list[list[float]],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add multiple vectors at once."""
        self._ensure_index()
        
        if metadata_list is None:
            metadata_list = [{}] * len(vectors)
        
        with self._lock:
            start_idx = len(self._vectors)
            indices = list(range(start_idx, start_idx + len(vectors)))
            
            self._vectors.extend(vectors)
            self._metadata.extend(metadata_list)
            
            if HAS_FAISS and self._index != "numpy_fallback":
                if not self._trained and len(self._vectors) >= self.nlist:
                    self._train_index()
                
                if self._trained:
                    vecs_np = np.array(vectors, dtype=np.float32)
                    self._index.add(vecs_np)
            
            return indices
    
    def _train_index(self) -> None:
        """Train the FAISS IVF index."""
        if not HAS_FAISS or self._trained:
            return
        
        logger.info("Training FAISS index on %d vectors", len(self._vectors))
        train_vecs = np.array(self._vectors, dtype=np.float32)
        self._index.train(train_vecs)
        self._index.add(train_vecs)
        self._trained = True
        logger.info("FAISS index trained and populated")
    
    def search(
        self,
        query: list[float],
        k: int = 10,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results
            
        Returns:
            List of (index, distance, metadata) tuples
        """
        self._ensure_index()
        
        if not self._vectors:
            return []
        
        k = min(k, len(self._vectors))
        
        if HAS_FAISS and self._trained and self._index != "numpy_fallback":
            # FAISS search
            query_np = np.array([query], dtype=np.float32)
            distances, indices = self._index.search(query_np, k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx >= 0 and idx < len(self._metadata):
                    results.append((int(idx), float(dist), self._metadata[idx]))
            return results
        else:
            # Numpy brute-force fallback
            return self._numpy_search(query, k)
    
    def _numpy_search(self, query: list[float], k: int) -> list[tuple[int, float, dict[str, Any]]]:
        """Brute-force search using numpy/pure Python."""
        if HAS_NUMPY:
            query_np = np.array(query, dtype=np.float32)
            vecs_np = np.array(self._vectors, dtype=np.float32)
            
            # Cosine similarity (assuming L2 normalized)
            similarities = np.dot(vecs_np, query_np)
            top_k = np.argsort(similarities)[::-1][:k]
            
            return [
                (int(idx), float(1 - similarities[idx]), self._metadata[idx])
                for idx in top_k
            ]
        else:
            # Pure Python fallback
            def cosine(a: list[float], b: list[float]) -> float:
                return sum(x * y for x, y in zip(a, b, strict=False))
            
            scored = [
                (i, 1 - cosine(query, vec), meta)
                for i, (vec, meta) in enumerate(zip(self._vectors, self._metadata, strict=False))
            ]
            scored.sort(key=lambda x: x[1])
            return scored[:k]
    
    def size(self) -> int:
        """Return number of vectors in index."""
        return len(self._vectors)
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        data = {
            "vectors": self._vectors,
            "metadata": self._metadata,
            "dim": self.dim,
            "nlist": self.nlist,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved FAISS index to %s", path)
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        with open(path) as f:
            data = json.load(f)
        
        self._vectors = data["vectors"]
        self._metadata = data["metadata"]
        self.dim = data.get("dim", self.dim)
        self.nlist = data.get("nlist", self.nlist)
        
        # Rebuild FAISS index
        self._index = None
        self._trained = False
        self._ensure_index()
        
        if HAS_FAISS and len(self._vectors) >= self.nlist:
            self._train_index()
        
        logger.info("Loaded FAISS index from %s (%d vectors)", path, len(self._vectors))


# Global instances (dictionary holder avoids 'global' statement)
_embedder_holder: dict[str, CodeBERTEmbedder] = {}
_embedder_lock = threading.Lock()


def get_codebert_embedder(config: EmbeddingConfig | None = None) -> CodeBERTEmbedder:
    """Get the global CodeBERT embedder instance."""
    with _embedder_lock:
        if "instance" not in _embedder_holder:
            _embedder_holder["instance"] = CodeBERTEmbedder(config)
        return _embedder_holder["instance"]


def embed_code(text: str) -> list[float]:
    """Convenience function to embed code/text using CodeBERT."""
    return get_codebert_embedder().embed(text)


def embed_code_batch(texts: list[str]) -> list[list[float]]:
    """Convenience function to batch embed code/text."""
    return get_codebert_embedder().embed_batch(texts)
