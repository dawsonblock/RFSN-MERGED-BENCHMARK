"""Patch-specific search strategies."""
from __future__ import annotations
from typing import List, Callable, Any
from .beam import BeamSearch


def search_patches(
    plan: Any,
    patch_generator: Callable[[Any], List[Any]],
    width: int = 3,
) -> List[Any]:
    """
    Search for patch candidates using beam search.
    
    Args:
        plan: The repair plan to generate patches for
        patch_generator: Function that generates patch candidates from a plan
        width: Beam width (number of candidates to explore)
        
    Returns:
        List of patch candidates to try
    """
    beam = BeamSearch(width=width)
    candidates = [plan]

    def expand_fn(p: Any) -> List[Any]:
        return patch_generator(p)

    return beam.expand(candidates, expand_fn)


def iterative_refinement(
    initial_patch: Any,
    refine_fn: Callable[[Any, str], Any],
    test_fn: Callable[[Any], tuple[bool, str]],
    max_iterations: int = 3,
) -> Any | None:
    """
    Iteratively refine a patch based on test feedback.
    
    Args:
        initial_patch: The starting patch
        refine_fn: Function to refine a patch given feedback
        test_fn: Function to test a patch, returns (passed, feedback)
        max_iterations: Maximum refinement iterations
        
    Returns:
        A passing patch, or None if max iterations exceeded
    """
    current = initial_patch
    
    for _ in range(max_iterations):
        passed, feedback = test_fn(current)
        if passed:
            return current
        
        # Refine based on feedback
        current = refine_fn(current, feedback)
    
    return None
