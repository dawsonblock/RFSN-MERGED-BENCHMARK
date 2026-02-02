"""Planner Layer v2 - Memory Adapter.

Read-only integration with ActionOutcomeStore for biasing decompositions
toward patterns that worked before.

The planner can READ historical outcomes but CANNOT write to memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfsn_controller.action_outcome_memory import ActionOutcomeStore


@dataclass
class DecompositionPrior:
    """A prior decomposition pattern from memory.

    Represents a previously successful decomposition pattern
    with its success metrics.
    """

    goal_type: str
    step_pattern: list[str]
    success_rate: float
    weight: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal_type": self.goal_type,
            "step_pattern": self.step_pattern,
            "success_rate": self.success_rate,
            "weight": self.weight,
            "n": self.n,
        }


class MemoryAdapter:
    """Read-only adapter for querying historical outcomes.

    The planner can READ historical outcomes to bias toward
    decompositions that worked before. It CANNOT write to memory.

    Ranking formula:
        score = success_rate * decay_weight * similarity

    Overfitting prevention:
        - Exponential decay with configurable half-life (default 14 days)
        - Similarity threshold: ignore outcomes with similarity < 0.25
        - Max age: prune outcomes older than 90 days
    """

    def __init__(
        self,
        memory_store: ActionOutcomeStore | None = None,
        min_similarity: float = 0.25,
    ):
        """Initialize the memory adapter.

        Args:
            memory_store: Optional ActionOutcomeStore for historical queries.
            min_similarity: Minimum similarity threshold for matches.
        """
        self._store = memory_store
        self._min_similarity = min_similarity

    def query_decomposition_priors(
        self,
        goal_type: str,
        repo_type: str,
        language: str,
        top_k: int = 5,
    ) -> list[DecompositionPrior]:
        """Query past outcomes to bias toward successful decompositions.

        Strategy:
        1. Filter by goal_type (repair, feature, refactor)
        2. Filter by repo_type and language
        3. Apply decay based on age
        4. Rank by success_rate * weight
        5. Return top-K patterns

        Args:
            goal_type: Type of goal (repair, feature, refactor).
            repo_type: Repository type identifier.
            language: Primary language.
            top_k: Maximum patterns to return.

        Returns:
            List of decomposition priors, sorted by weight.
        """
        if self._store is None:
            return []

        # Query the ActionOutcomeStore for matching patterns
        # This integrates with the existing memory system
        try:
            from rfsn_controller.action_outcome_memory import (
                make_context_signature,
            )

            context = make_context_signature(
                failure_class=goal_type,
                repo_type=repo_type,
                language=language,
                env={},
                attempt_count=0,
                failing_test_file=None,
                sig=None,
                stalled=False,
            )

            priors = self._store.query_action_priors(
                context,
                top_k=top_k,
                min_similarity=self._min_similarity,
            )

            # Convert to DecompositionPrior format
            results = []
            for prior in priors:
                results.append(
                    DecompositionPrior(
                        goal_type=goal_type,
                        step_pattern=[prior.action_key],
                        success_rate=prior.success_rate,
                        weight=prior.weight,
                        n=prior.n,
                    )
                )
            return results

        except ImportError:
            return []
        except Exception:
            return []

    def get_similarity_score(
        self,
        goal: str,
        historical_goal: str,
    ) -> float:
        """Calculate similarity between goals.

        Uses Jaccard similarity on tokens for determinism.

        Args:
            goal: Current goal description.
            historical_goal: Historical goal to compare.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        tokens_a = set(goal.lower().split())
        tokens_b = set(historical_goal.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    def has_memory(self) -> bool:
        """Check if memory store is available.

        Returns:
            True if memory store is configured.
        """
        return self._store is not None
