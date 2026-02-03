"""Ensemble patcher adapter - bridges 3-planner ensemble to episode runner.

This adapter provides a unified interface for using the multi-planner
ensemble mode alongside the standard single-LLM approach.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from swebench_max.candidate_generator import (
    generate_candidates, 
    record_candidate_result,
    GeneratorState,
)
from swebench_max.candidate import Candidate

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePatchResult:
    """Result from ensemble patch generation."""
    patch_text: str
    summary: str
    metadata: dict[str, Any]


class EnsemblePatcher:
    """
    Wrapper around the 3-planner ensemble candidate generator.
    
    Features:
    - Primary, Alt, Skeptic planners
    - Automatic patch deduplication
    - Failure fingerprint retrieval
    - Cross-attempt learning
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._state = GeneratorState()
    
    def reset(self) -> None:
        """Reset state for a new task."""
        self._state = GeneratorState()
    
    def generate(
        self, 
        task: dict[str, Any], 
        repo_root: str,
    ) -> list[EnsemblePatchResult]:
        """
        Generate patch candidates using 3-planner ensemble.
        
        Args:
            task: SWE-bench task dict
            repo_root: Path to repository
            
        Returns:
            List of patch results (typically 1-3 from different planners)
        """
        # Call the underlying generator
        candidates = generate_candidates(
            issue=task,
            cfg=self.config,
            state=self._state,
            repo_root=repo_root,
        )
        
        logger.info("Ensemble generated %d candidates", len(candidates))
        
        # Convert to our result format
        results = []
        for cand in candidates:
            results.append(EnsemblePatchResult(
                patch_text=cand.patch,
                summary=f"[{cand.meta.get('planner', 'unknown')}] Ensemble patch",
                metadata={
                    "variant": "ensemble",
                    "planner": cand.meta.get("planner", "unknown"),
                    "source": cand.meta.get("source", "llm"),
                    "key": cand.key,
                },
            ))
        
        # Increment round for next call
        self._state.round_idx += 1
        
        return results
    
    def record_result(
        self,
        candidate: Candidate,
        success: bool,
        task: dict[str, Any],
    ) -> None:
        """Record the result of a candidate for learning."""
        record_candidate_result(
            candidate=candidate,
            success=success,
            state=self._state,
            issue=task,
            cfg=self.config,
        )


# Singleton instance for easy access
_ensemble_patcher: EnsemblePatcher | None = None


def get_ensemble_patcher(config: dict[str, Any] | None = None) -> EnsemblePatcher:
    """Get or create the ensemble patcher singleton."""
    global _ensemble_patcher
    if _ensemble_patcher is None or config is not None:
        _ensemble_patcher = EnsemblePatcher(config)
    return _ensemble_patcher
