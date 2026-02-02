"""Unified multi-layer localization interface.

Combines:
1. Trace parsing (high confidence)
2. Ripgrep lexical search (fast)
3. Embedding semantic search (deep understanding)
4. Symbol/import graph analysis (structural)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .trace import parse_python_traceback
from .ripgrep import localize_from_issue, localize_from_trace, RipgrepConfig
from .types import LocalizationHit
from .symbol_index import SymbolIndex, build_symbol_index

try:
    from .embeddings import localize_semantic, EmbeddingConfig, HAS_SEMANTIC
except ImportError:
    HAS_SEMANTIC = False
    localize_semantic = None
    EmbeddingConfig = None

from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class LocalizationConfig:
    """Configuration for multi-layer localization."""
    
    # Layer enablement
    use_trace: bool = True
    use_ripgrep: bool = True
    use_embeddings: bool = True
    use_symbols: bool = True
    
    # Layer-specific configs
    ripgrep_config: Optional[RipgrepConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    
    # Result fusion
    max_results: int = 50
    score_weights: dict = None
    
    def __post_init__(self):
        if self.score_weights is None:
            self.score_weights = {
                "trace": 1.0,        # Highest confidence
                "symbol": 0.9,       # High confidence for definitions
                "ripgrep": 0.7,      # Medium confidence for keyword matches
                "embedding": 0.8,    # Good for semantic understanding
            }
        
        if self.ripgrep_config is None:
            self.ripgrep_config = RipgrepConfig()
        
        if self.embedding_config is None and EmbeddingConfig:
            self.embedding_config = EmbeddingConfig()


class MultiLayerLocalizer:
    """Multi-layer localization combining all strategies."""
    
    def __init__(self, config: LocalizationConfig = None):
        """Initialize localizer.
        
        Args:
            config: Localization configuration
        """
        self.config = config or LocalizationConfig()
        self.symbol_index: Optional[SymbolIndex] = None
    
    def localize(
        self,
        problem_statement: str,
        repo_dir: Path,
        traceback: Optional[str] = None,
        failing_tests: Optional[List[str]] = None,
    ) -> List[LocalizationHit]:
        """Perform multi-layer localization.
        
        Args:
            problem_statement: Issue description
            repo_dir: Repository root directory
            traceback: Optional stack trace
            failing_tests: Optional list of failing test names
            
        Returns:
            Ranked list of localization hits
        """
        all_hits = []
        
        # Layer 1: Trace parsing (highest confidence)
        if self.config.use_trace and traceback:
            logger.info("Layer 1: Parsing stack trace")
            trace_hits = parse_python_traceback(traceback, repo_dir)
            for hit in trace_hits:
                hit.score *= self.config.score_weights.get("trace", 1.0)
            all_hits.extend(trace_hits)
            logger.info(f"Found {len(trace_hits)} trace hits")
        
        # Layer 2a: Ripgrep lexical search (fast)
        if self.config.use_ripgrep:
            logger.info("Layer 2a: Ripgrep lexical search")
            
            # Search from issue
            ripgrep_hits = localize_from_issue(
                problem_statement,
                repo_dir,
                self.config.ripgrep_config,
            )
            
            # Also search from trace if available
            if traceback:
                ripgrep_hits.extend(localize_from_trace(
                    traceback,
                    repo_dir,
                    self.config.ripgrep_config,
                ))
            
            # Weight scores
            for hit in ripgrep_hits:
                hit.score *= self.config.score_weights.get("ripgrep", 0.7)
            
            all_hits.extend(ripgrep_hits)
            logger.info(f"Found {len(ripgrep_hits)} ripgrep hits")
        
        # Layer 2b: Symbol/import graph (structural)
        if self.config.use_symbols:
            logger.info("Layer 2b: Symbol index search")
            
            # Build index if not already built
            if self.symbol_index is None:
                self.symbol_index = build_symbol_index(repo_dir)
            
            # Extract potential symbol names from issue
            import re
            identifiers = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', problem_statement)  # CamelCase
            identifiers += re.findall(r'\b[a-z]+_[a-z_]+\b', problem_statement)  # snake_case
            
            symbol_hits = []
            for identifier in identifiers[:20]:  # Limit to top 20
                symbol_hits.extend(self.symbol_index.localize_by_symbol(identifier))
            
            # Weight scores
            for hit in symbol_hits:
                hit.score *= self.config.score_weights.get("symbol", 0.9)
            
            all_hits.extend(symbol_hits)
            logger.info(f"Found {len(symbol_hits)} symbol hits")
        
        # Layer 3: Embedding semantic search (deep)
        if self.config.use_embeddings and HAS_SEMANTIC:
            logger.info("Layer 3: Embedding semantic search")
            
            try:
                # Build query from problem + trace
                query_parts = [problem_statement]
                if traceback:
                    query_parts.append(f"Stack trace:\n{traceback}")
                query = "\n\n".join(query_parts)
                
                embedding_hits = localize_semantic(
                    query,
                    repo_dir,
                    self.config.embedding_config,
                )
                
                # Weight scores
                for hit in embedding_hits:
                    hit.score *= self.config.score_weights.get("embedding", 0.8)
                
                all_hits.extend(embedding_hits)
                logger.info(f"Found {len(embedding_hits)} embedding hits")
                
            except Exception as e:
                logger.warning(f"Embedding search failed: {e}")
        elif self.config.use_embeddings:
            logger.warning("Embedding search requested but dependencies not available")
        
        # Fuse and rank results
        final_hits = self._fuse_hits(all_hits)
        
        logger.info(f"Total unique hits after fusion: {len(final_hits)}")
        
        return final_hits[:self.config.max_results]
    
    def _fuse_hits(self, hits: List[LocalizationHit]) -> List[LocalizationHit]:
        """Fuse and deduplicate hits from multiple layers.
        
        Args:
            hits: List of all hits from all layers
            
        Returns:
            Fused and ranked list of hits
        """
        # Group by file and line range
        grouped = {}
        
        for hit in hits:
            key = (hit.file_path, hit.line_start, hit.line_end)
            
            if key not in grouped:
                grouped[key] = hit
            else:
                # Merge: take max score, combine evidence
                existing = grouped[key]
                existing.score = max(existing.score, hit.score)
                
                if hit.evidence not in existing.evidence:
                    existing.evidence += f"; {hit.evidence}"
                
                if hit.method not in existing.method:
                    existing.method += f"+{hit.method}"
        
        # Sort by score
        fused = list(grouped.values())
        fused.sort(key=lambda h: h.score, reverse=True)
        
        return fused


def localize_issue(
    problem_statement: str,
    repo_dir: Path,
    traceback: Optional[str] = None,
    failing_tests: Optional[List[str]] = None,
    config: LocalizationConfig = None,
) -> List[LocalizationHit]:
    """High-level API for issue localization.
    
    This is the main entry point for localization. It runs all available
    layers and returns ranked, fused results.
    
    Args:
        problem_statement: Issue description text
        repo_dir: Repository root directory
        traceback: Optional stack trace
        failing_tests: Optional list of failing test names
        config: Optional localization configuration
        
    Returns:
        Ranked list of localization hits
    """
    localizer = MultiLayerLocalizer(config)
    return localizer.localize(
        problem_statement,
        repo_dir,
        traceback,
        failing_tests,
    )
