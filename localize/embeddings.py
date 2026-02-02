"""Embedding-based semantic localization layer.

Uses sentence embeddings + FAISS for semantic similarity search across repo chunks.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    np = None
    faiss = None
    SentenceTransformer = None

from .types import LocalizationHit
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    """A chunk of code for embedding."""
    
    file_path: str
    line_start: int
    line_end: int
    content: str
    chunk_type: str = "function"  # function, class, module
    
    def to_text(self) -> str:
        """Convert to text for embedding."""
        return f"{self.file_path}:{self.line_start}-{self.line_end}\n{self.content}"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding-based localization."""
    
    # Model
    model_name: str = "all-MiniLM-L6-v2"  # Fast, small model
    cache_dir: Optional[Path] = None
    
    # Chunking
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64
    
    # Search
    top_k: int = 20
    min_similarity: float = 0.5


class EmbeddingLocalizer:
    """Semantic localization using embeddings."""
    
    def __init__(self, config: EmbeddingConfig = None):
        """Initialize embedding localizer.
        
        Args:
            config: Embedding configuration
        """
        if not HAS_SEMANTIC:
            raise ImportError(
                "Semantic localization requires: pip install sentence-transformers faiss-cpu"
            )
        
        self.config = config or EmbeddingConfig()
        
        # Load model
        logger.info(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)
        
        # Index state
        self.index: Optional[faiss.Index] = None
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def build_index(self, repo_dir: Path, cache_path: Optional[Path] = None) -> None:
        """Build FAISS index for repository.
        
        Args:
            repo_dir: Repository root directory
            cache_path: Optional path to cache index
        """
        # Check cache
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached index from {cache_path}")
            self._load_index(cache_path)
            return
        
        logger.info(f"Building embedding index for {repo_dir}")
        
        # Extract chunks
        self.chunks = self._extract_chunks(repo_dir)
        logger.info(f"Extracted {len(self.chunks)} code chunks")
        
        if not self.chunks:
            logger.warning("No chunks extracted, index will be empty")
            return
        
        # Generate embeddings
        chunk_texts = [chunk.to_text() for chunk in self.chunks]
        logger.info("Generating embeddings...")
        self.embeddings = self.model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
        # Save cache
        if cache_path:
            self._save_index(cache_path)
    
    def search(self, query: str, top_k: int = None) -> List[LocalizationHit]:
        """Search for semantically similar code chunks.
        
        Args:
            query: Query text (issue description, error message, etc.)
            top_k: Number of results to return
            
        Returns:
            List of localization hits
        """
        if self.index is None or not self.chunks:
            logger.warning("Index not built, cannot search")
            return []
        
        if top_k is None:
            top_k = self.config.top_k
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Convert to hits
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if score < self.config.min_similarity:
                continue
            
            chunk = self.chunks[idx]
            hit = LocalizationHit(
                file_path=chunk.file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                score=float(score),
                evidence=f"Semantic similarity: {score:.3f} ({chunk.chunk_type})",
                method="embedding_semantic",
            )
            hits.append(hit)
        
        logger.info(f"Found {len(hits)} semantic hits for query")
        
        return hits
    
    def _extract_chunks(self, repo_dir: Path) -> List[CodeChunk]:
        """Extract code chunks from repository.
        
        Args:
            repo_dir: Repository root directory
            
        Returns:
            List of code chunks
        """
        chunks = []
        
        # Find Python files
        for py_file in repo_dir.rglob("*.py"):
            # Skip common ignored directories
            if any(part in [".git", "__pycache__", "venv", ".venv", "build", "dist"]
                   for part in py_file.parts):
                continue
            
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Simple chunking by functions/classes
                file_chunks = self._chunk_file(
                    str(py_file.relative_to(repo_dir)),
                    content,
                )
                chunks.extend(file_chunks)
                
            except Exception as e:
                logger.debug(f"Failed to process {py_file}: {e}")
                continue
        
        return chunks
    
    def _chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk a single file into embedable units.
        
        Args:
            file_path: Relative file path
            content: File content
            
        Returns:
            List of code chunks
        """
        chunks = []
        lines = content.splitlines()
        
        # Simple heuristic: chunk by class/function definitions
        current_chunk_start = 0
        current_chunk_lines = []
        chunk_type = "module"
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Detect class/function start
            if stripped.startswith("class ") or stripped.startswith("def "):
                # Save previous chunk if substantial
                if len(current_chunk_lines) > 5:
                    chunk = CodeChunk(
                        file_path=file_path,
                        line_start=current_chunk_start + 1,
                        line_end=i,
                        content="\n".join(current_chunk_lines),
                        chunk_type=chunk_type,
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk_start = i
                current_chunk_lines = [line]
                chunk_type = "class" if stripped.startswith("class") else "function"
            else:
                current_chunk_lines.append(line)
            
            # Break if chunk gets too large
            if len("\n".join(current_chunk_lines)) > self.config.chunk_size * 4:
                if current_chunk_lines:
                    chunk = CodeChunk(
                        file_path=file_path,
                        line_start=current_chunk_start + 1,
                        line_end=i + 1,
                        content="\n".join(current_chunk_lines),
                        chunk_type=chunk_type,
                    )
                    chunks.append(chunk)
                    current_chunk_start = i + 1
                    current_chunk_lines = []
                    chunk_type = "module"
        
        # Add final chunk
        if current_chunk_lines:
            chunk = CodeChunk(
                file_path=file_path,
                line_start=current_chunk_start + 1,
                line_end=len(lines),
                content="\n".join(current_chunk_lines),
                chunk_type=chunk_type,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _save_index(self, cache_path: Path) -> None:
        """Save index to disk.
        
        Args:
            cache_path: Path to save index
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(cache_path) + ".index")
        
        # Save chunks and embeddings
        with open(str(cache_path) + ".pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings,
            }, f)
        
        logger.info(f"Saved index to {cache_path}")
    
    def _load_index(self, cache_path: Path) -> None:
        """Load index from disk.
        
        Args:
            cache_path: Path to load index from
        """
        # Load FAISS index
        self.index = faiss.read_index(str(cache_path) + ".index")
        
        # Load chunks and embeddings
        with open(str(cache_path) + ".pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
        
        logger.info(f"Loaded index from {cache_path}")


def localize_semantic(
    query: str,
    repo_dir: Path,
    config: EmbeddingConfig = None,
    cache_path: Optional[Path] = None,
) -> List[LocalizationHit]:
    """Perform semantic localization.
    
    Args:
        query: Query text
        repo_dir: Repository root directory
        config: Embedding configuration
        cache_path: Optional cache path
        
    Returns:
        List of localization hits
    """
    if not HAS_SEMANTIC:
        logger.warning("Semantic localization not available, install dependencies")
        return []
    
    localizer = EmbeddingLocalizer(config)
    localizer.build_index(repo_dir, cache_path)
    return localizer.search(query)
