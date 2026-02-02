"""Multi-step beam search over patch space.

Explores multiple patch candidates in parallel, using git rollback
to safely test hypotheses without polluting repository state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..git_rollback import GitRollbackManager


@dataclass
class Candidate:
    """A patch candidate in the beam search tree."""
    
    candidate_id: str
    patch_diff: str
    score: float
    depth: int
    parent_id: str | None = None
    snapshot_id: str | None = None
    test_result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if this candidate fully solves the problem."""
        return self.score >= 0.95


@dataclass
class BeamSearchConfig:
    """Configuration for beam search."""
    
    beam_width: int = 3          # Number of candidates to keep
    max_depth: int = 5           # Maximum expansion steps
    score_threshold: float = 0.95  # Score to terminate early
    min_improvement: float = 0.05  # Minimum score gain to continue
    timeout_seconds: float = 300.0  # Total search timeout
    patches_per_step: int = 3    # Patches to generate per expansion


@dataclass  
class BeamSearchResult:
    """Result of beam search."""
    
    success: bool
    best_candidate: Candidate | None
    all_candidates: list[Candidate]
    search_stats: dict[str, Any]


class BeamSearcher:
    """Multi-step beam search over patch space.
    
    Explores multiple repair hypotheses in parallel, keeping the
    top-k scoring candidates at each step.
    
    Key features:
    - Git-based rollback for safe exploration
    - Configurable beam width and depth
    - Early termination on success
    - Score-based pruning
    
    Example:
        searcher = BeamSearcher(config, rollback_mgr)
        result = searcher.search(
            repo_path="/path/to/repo",
            goal="Fix failing test_foo",
            generate_fn=my_patch_generator,
            evaluate_fn=my_test_runner,
        )
        if result.success:
            apply_patch(result.best_candidate.patch_diff)
    """
    
    def __init__(
        self,
        config: BeamSearchConfig | None = None,
        rollback_mgr: GitRollbackManager | None = None,
    ):
        """Initialize beam searcher.
        
        Args:
            config: Search configuration.
            rollback_mgr: Git rollback manager (created if not provided).
        """
        self.config = config or BeamSearchConfig()
        self.rollback = rollback_mgr or GitRollbackManager()
        self._candidate_counter = 0
    
    def _generate_candidate_id(self) -> str:
        """Generate a unique candidate ID."""
        self._candidate_counter += 1
        return f"c{self._candidate_counter:04d}"
    
    def search(  # noqa: PLR0912
        self,
        repo_path: str,
        goal: str,
        generate_fn: Callable[[str, str, int], list[str]],
        evaluate_fn: Callable[[str, str], tuple[float, dict]],
        context: dict[str, Any] | None = None,
    ) -> BeamSearchResult:
        """Run beam search to find best patch sequence.
        
        Args:
            repo_path: Path to the git repository.
            goal: Description of what to fix/achieve.
            generate_fn: Function(repo_path, goal, n) -> list[patch_diffs]
            evaluate_fn: Function(repo_path, patch) -> (score, test_result)
            context: Optional additional context.
            
        Returns:
            BeamSearchResult with best candidate and stats.
        """
        start_time = time.time()
        all_candidates: list[Candidate] = []
        stats = {
            "total_candidates": 0,
            "total_evaluations": 0,
            "depths_explored": 0,
            "early_termination": False,
            "timeout": False,
        }
        
        # Save initial state
        initial_snapshot = self.rollback.save_working_state(repo_path)
        
        try:
            # Initialize beam with empty candidate (current state)
            beam: list[Candidate] = [
                Candidate(
                    candidate_id=self._generate_candidate_id(),
                    patch_diff="",
                    score=0.0,
                    depth=0,
                    snapshot_id=initial_snapshot.snapshot_id,
                )
            ]
            
            best_candidate: Candidate | None = None
            
            for depth in range(self.config.max_depth):
                stats["depths_explored"] = depth + 1
                
                # Check timeout
                if time.time() - start_time > self.config.timeout_seconds:
                    stats["timeout"] = True
                    break
                
                # Expand each candidate in the beam
                new_candidates: list[Candidate] = []
                
                for parent in beam:
                    # Restore to parent state
                    if parent.snapshot_id:
                        self.rollback.restore(repo_path, parent.snapshot_id)
                    
                    # Apply parent's patch if any
                    if parent.patch_diff:
                        self._apply_patch(repo_path, parent.patch_diff)
                    
                    # Generate new patch candidates
                    patches = generate_fn(
                        repo_path,
                        goal,
                        self.config.patches_per_step,
                    )
                    
                    for patch in patches:
                        if not patch.strip():
                            continue
                        
                        # Create snapshot before applying
                        snap = self.rollback.create_snapshot(
                            repo_path,
                            f"depth{depth}_candidate",
                        )
                        
                        # Apply and evaluate
                        self._apply_patch(repo_path, patch)
                        score, test_result = evaluate_fn(repo_path, patch)
                        stats["total_evaluations"] += 1
                        
                        candidate = Candidate(
                            candidate_id=self._generate_candidate_id(),
                            patch_diff=self._combine_patches(
                                parent.patch_diff, patch
                            ),
                            score=score,
                            depth=depth + 1,
                            parent_id=parent.candidate_id,
                            snapshot_id=snap.snapshot_id,
                            test_result=test_result,
                        )
                        
                        new_candidates.append(candidate)
                        all_candidates.append(candidate)
                        stats["total_candidates"] += 1
                        
                        # Check for success
                        if candidate.is_success:
                            stats["early_termination"] = True
                            best_candidate = candidate
                            break
                        
                        # Rollback for next candidate
                        self.rollback.restore(repo_path, snap.snapshot_id)
                    
                    if stats["early_termination"]:
                        break
                
                if stats["early_termination"]:
                    break
                
                if not new_candidates:
                    break
                
                # Prune to top-k
                new_candidates.sort(key=lambda c: c.score, reverse=True)
                beam = new_candidates[:self.config.beam_width]
                
                # Update best
                if beam and (not best_candidate or beam[0].score > best_candidate.score):
                    best_candidate = beam[0]
                
                # Check if making progress
                if best_candidate and best_candidate.score >= self.config.score_threshold:
                    stats["early_termination"] = True
                    break
        
        finally:
            # Restore initial state
            self.rollback.restore_working_state(repo_path, initial_snapshot)
            
            # Cleanup beam search snapshots
            self.rollback.cleanup_all(repo_path)
        
        return BeamSearchResult(
            success=best_candidate is not None and best_candidate.is_success,
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            search_stats=stats,
        )
    
    def _apply_patch(self, repo_path: str, patch: str) -> bool:
        """Apply a patch to the repository.
        
        Args:
            repo_path: Path to the repository.
            patch: Unified diff patch content.
            
        Returns:
            True if applied successfully.
        """
        import subprocess
        
        if not patch.strip():
            return True
        
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "apply", "--3way", "-"],
                input=patch,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _combine_patches(self, parent_patch: str, child_patch: str) -> str:
        """Combine parent and child patches.
        
        Args:
            parent_patch: Accumulated patch from parent.
            child_patch: New patch to add.
            
        Returns:
            Combined patch content.
        """
        if not parent_patch:
            return child_patch
        if not child_patch:
            return parent_patch
        return f"{parent_patch}\n{child_patch}"


def score_test_result(
    test_result: dict[str, Any],
    diff_lines: int = 0,
) -> float:
    """Score a test result for beam search ranking.
    
    Args:
        test_result: Dict with test execution results.
        diff_lines: Number of lines changed.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0
    
    # Primary signal: test pass rate
    if test_result.get("all_pass"):
        score = 1.0
    elif "pass_rate" in test_result:
        score = test_result["pass_rate"] * 0.8
    elif "passed" in test_result and "total" in test_result:
        total = test_result["total"]
        if total > 0:
            score = (test_result["passed"] / total) * 0.8
    
    # Penalty for new failures (regressions)
    new_failures = test_result.get("new_failures", 0)
    score -= 0.15 * new_failures
    
    # Penalty for large diffs
    if diff_lines > 0:
        score -= 0.02 * (diff_lines / 100)
    
    # Bonus for fixing the focused test
    if test_result.get("focused_test_pass"):
        score += 0.1
    
    return max(0.0, min(1.0, score))


def create_beam_searcher(
    beam_width: int = 3,
    max_depth: int = 5,
    score_threshold: float = 0.95,
    timeout_seconds: float = 300.0,
) -> BeamSearcher:
    """Create a configured beam searcher.
    
    Args:
        beam_width: Number of candidates to keep per step.
        max_depth: Maximum expansion depth.
        score_threshold: Score to terminate early.
        timeout_seconds: Total search timeout.
        
    Returns:
        Configured BeamSearcher instance.
    """
    config = BeamSearchConfig(
        beam_width=beam_width,
        max_depth=max_depth,
        score_threshold=score_threshold,
        timeout_seconds=timeout_seconds,
    )
    return BeamSearcher(config=config)
