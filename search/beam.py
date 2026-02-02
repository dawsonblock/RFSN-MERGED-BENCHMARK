"""Beam search for patch exploration."""
from __future__ import annotations
from typing import List, Callable, Any, TypeVar

T = TypeVar("T")


class BeamSearch:
    """
    Beam search over patch candidates.
    
    Maintains a fixed-width beam of candidates and expands
    the most promising ones at each step.
    """
    
    def __init__(self, width: int = 4):
        self.width = width

    def expand(
        self, 
        candidates: List[T], 
        expand_fn: Callable[[T], List[T]],
        score_fn: Callable[[T], float] | None = None,
    ) -> List[T]:
        """
        Expand candidates using expand_fn, keeping top-width by score.
        
        Args:
            candidates: Current beam of candidates
            expand_fn: Function to expand a candidate into multiple new candidates
            score_fn: Optional scoring function (higher = better). If None, keeps first width.
            
        Returns:
            Top-width expanded candidates
        """
        expanded: List[T] = []
        for c in candidates:
            expanded.extend(expand_fn(c))
        
        if score_fn is not None:
            expanded.sort(key=score_fn, reverse=True)
        
        return expanded[:self.width]
    
    def search(
        self,
        initial: List[T],
        expand_fn: Callable[[T], List[T]],
        is_goal: Callable[[T], bool],
        score_fn: Callable[[T], float] | None = None,
        max_depth: int = 5,
    ) -> T | None:
        """
        Full beam search to find a goal state.
        
        Args:
            initial: Initial candidates
            expand_fn: Function to expand candidates
            is_goal: Function to check if a candidate is a goal state
            score_fn: Optional scoring function
            max_depth: Maximum search depth
            
        Returns:
            First goal candidate found, or None
        """
        beam = initial[:self.width]
        
        for _ in range(max_depth):
            # Check for goal states
            for candidate in beam:
                if is_goal(candidate):
                    return candidate
            
            # Expand beam
            beam = self.expand(beam, expand_fn, score_fn)
            
            if not beam:
                break
        
        return None
