"""Basic orchestrator loop - simple integration example."""
from __future__ import annotations

from typing import Dict, Any, Callable, List
from planner.planner import generate_plan
from search.patch_search import search_patches
from learning.bandit import StrategyBandit
from learning.outcomes import Outcome, score
from memory.store import MemoryStore

# Global state (for basic usage)
bandit = StrategyBandit()
memory = MemoryStore()


def run_episode(
    task: Dict[str, Any],
    patch_generator: Callable[[Any], List[Any]],
    executor: Callable[[Any], Outcome],
) -> bool:
    """
    Run a single repair episode.
    
    This is the basic orchestrator loop that ties together:
    - Memory retrieval
    - Planning
    - Patch search
    - Execution
    - Learning
    
    Args:
        task: Task dict with repo, description, failing_files, etc.
        patch_generator: Function to generate patch candidates from a plan
        executor: Function to execute a patch and return Outcome
        
    Returns:
        True if task was solved, False otherwise
    """
    repo = task.get("repo", "unknown")
    
    # Retrieve from memory
    retrieved = memory.retrieve(repo)
    
    # Generate plan
    plan = generate_plan(task, retrieved)
    
    # Search for patch candidates
    candidates = search_patches(plan, patch_generator)
    
    # Execute candidates
    for patch in candidates:
        result = executor(patch)
        reward = score(result)
        bandit.update("default", reward)
        
        # Record to memory
        memory.add({
            "repo": repo,
            "passed": result.passed,
            "reward": reward,
            "tag": repo,
        })
        
        if result.passed:
            return True
    
    return False
