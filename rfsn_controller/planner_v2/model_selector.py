"""Model Arbitration - Learned Model Selection.

This module implements a Multi-Armed Bandit (Thompson Sampling) approach 
to selecting the best LLM for a given task context (language, failure type).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelOption:
    """Configuration for an available model."""
    name: str
    provider: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    context_window: int = 8192
    
    @property
    def id(self) -> str:
        return f"{self.provider}/{self.name}"


@dataclass
class ModelStats:
    """Statistics for a model in a specific context."""
    successes: int = 1  # Beta prior alpha=1
    failures: int = 1   # Beta prior beta=1
    latency_ms_sum: int = 0
    cost_sum: float = 0.0
    
    @property
    def total_attempts(self) -> int:
        return self.successes + self.failures - 2  # Subtract priors
        
    @property
    def expected_value(self) -> float:
        """Mean of Beta distribution."""
        return self.successes / (self.successes + self.failures)
        
    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)."""
        return random.betavariate(self.successes, self.failures)


class ModelSelector:
    """Selects the best model using Thompson Sampling."""
    
    def __init__(self, storage_path: Path | None = None):
        self._storage_path = storage_path
        self._models: dict[str, ModelOption] = {}
        # Map context_key -> model_id -> ModelStats
        self._stats: dict[str, dict[str, ModelStats]] = {}
        
        self._register_defaults()
        if storage_path and storage_path.exists():
            self._load()

    def _register_defaults(self):
        """Register default models."""
        # This would normally come from configuration
        defaults = [
            ModelOption("gpt-4-turbo", "openai", 0.01, 0.03),
            ModelOption("claude-3-opus", "anthropic", 0.015, 0.075),
            ModelOption("claude-3-sonnet", "anthropic", 0.003, 0.015),
            ModelOption("gpt-3.5-turbo", "openai", 0.0005, 0.0015),
        ]
        for m in defaults:
            self.register_model(m)

    def register_model(self, model: ModelOption):
        """Register an available model."""
        self._models[model.id] = model

    def select_model(
        self,
        goal_type: str,
        failure_type: str,
        language: str,
        k: int = 1,
    ) -> list[ModelOption]:
        """Select top-k models using Thompson Sampling.
        
        Args:
            goal_type: Task goal type (repair, feature).
            failure_type: Failure classification.
            language: Programming language.
            k: Number of models to select.
            
        Returns:
            List of selected ModelOptions (best first).
        """
        context_key = f"{goal_type}:{failure_type}:{language}"
        
        # Ensure stats exist for this context
        if context_key not in self._stats:
            self._stats[context_key] = {
                mid: ModelStats() for mid in self._models.keys()
            }
        
        # Ensure all registered models have stats
        current_stats = self._stats[context_key]
        for mid in self._models:
            if mid not in current_stats:
                current_stats[mid] = ModelStats()
        
        # Thompson Sampling: Sample expected value for each arm
        scores = []
        for mid, model in self._models.items():
            stats = current_stats[mid]
            sample_score = stats.sample()
            
            # Penalize slightly for cost (simple heuristic)
            # Normalized cost penalty: expensive models need higher success rate
            cost_penalty = (model.cost_per_1k_input + model.cost_per_1k_output) * 10
            adjusted_score = sample_score - (cost_penalty * 0.1)
            
            scores.append((adjusted_score, model))
        
        # Sort by sampled score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [s[1] for s in scores[:k]]

    def record_outcome(
        self,
        model_id: str,
        goal_type: str,
        failure_type: str,
        language: str,
        success: bool,
        latency_ms: int = 0,
    ):
        """Record the outcome of a model usage.
        
        Args:
            model_id: ID of model used.
            goal_type: Task goal type.
            failure_type: Failure classification.
            language: Programming language.
            success: Whether task succeeded.
            latency_ms: Duration in ms.
        """
        context_key = f"{goal_type}:{failure_type}:{language}"
        
        if context_key not in self._stats:
            self._stats[context_key] = {}
        if model_id not in self._stats[context_key]:
            self._stats[context_key][model_id] = ModelStats()
            
        stats = self._stats[context_key][model_id]
        if success:
            stats.successes += 1
        else:
            stats.failures += 1
            
        stats.latency_ms_sum += latency_ms
        
        self._save()

    def _save(self):
        """Persist stats to storage."""
        if not self._storage_path:
            return
            
        data = {}
        for ctx, models in self._stats.items():
            data[ctx] = {
                mid: {
                    "successes": s.successes,
                    "failures": s.failures,
                    "latency_ms_sum": s.latency_ms_sum
                }
                for mid, s in models.items()
            }
            
        with open(self._storage_path, "w") as f:
            json.dump(data, f)
            
    def _load(self):
        """Load stats from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
            
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
                
            for ctx, models in data.items():
                self._stats[ctx] = {}
                for mid, s_dict in models.items():
                    self._stats[ctx][mid] = ModelStats(
                        successes=s_dict["successes"],
                        failures=s_dict["failures"],
                        latency_ms_sum=s_dict.get("latency_ms_sum", 0)
                    )
        except Exception:
            pass # Ignore corrupt cache
