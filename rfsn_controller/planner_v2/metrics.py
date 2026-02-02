"""Metrics - Planner performance metrics collection.

Provides instrumentation for tracking planner performance,
cache hit rates, revision counts, and execution timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannerMetrics:
    """Aggregate planner metrics."""
    
    # Plan generation
    plans_generated: int = 0
    plans_from_cache: int = 0
    plans_from_llm: int = 0
    plans_from_pattern: int = 0
    
    # Execution
    steps_executed: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    parallel_batches: int = 0
    parallel_steps: int = 0
    
    # Revision
    revisions_total: int = 0
    revisions_by_category: dict[str, int] = field(default_factory=dict)
    
    # QA
    qa_evaluations: int = 0
    qa_rejections: int = 0
    
    # Timing
    total_plan_time_ms: int = 0
    total_step_time_ms: int = 0
    avg_plan_time_ms: float = 0.0
    avg_step_time_ms: float = 0.0
    avg_steps_per_plan: float = 0.0
    
    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "plans_generated": self.plans_generated,
            "plans_from_cache": self.plans_from_cache,
            "plans_from_llm": self.plans_from_llm,
            "plans_from_pattern": self.plans_from_pattern,
            "steps_executed": self.steps_executed,
            "steps_succeeded": self.steps_succeeded,
            "steps_failed": self.steps_failed,
            "parallel_batches": self.parallel_batches,
            "parallel_steps": self.parallel_steps,
            "revisions_total": self.revisions_total,
            "revisions_by_category": self.revisions_by_category,
            "qa_evaluations": self.qa_evaluations,
            "qa_rejections": self.qa_rejections,
            "total_plan_time_ms": self.total_plan_time_ms,
            "total_step_time_ms": self.total_step_time_ms,
            "avg_plan_time_ms": self.avg_plan_time_ms,
            "avg_step_time_ms": self.avg_step_time_ms,
            "avg_steps_per_plan": self.avg_steps_per_plan,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "step_success_rate": self.step_success_rate,
        }
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def step_success_rate(self) -> float:
        return self.steps_succeeded / self.steps_executed if self.steps_executed > 0 else 0.0


class MetricsCollector:
    """Collects and manages planner metrics."""
    
    def __init__(self):
        self._metrics = PlannerMetrics()
        self._start_time = time.monotonic()
        self._plan_start: float | None = None
        self._step_start: float | None = None
    
    def record_plan_generated(
        self,
        source: str,
        step_count: int,
        time_ms: int,
    ) -> None:
        """Record a plan generation event.
        
        Args:
            source: Source of plan ("cache", "llm", "pattern")
            step_count: Number of steps in plan.
            time_ms: Time to generate.
        """
        self._metrics.plans_generated += 1
        self._metrics.total_plan_time_ms += time_ms
        
        if source == "cache":
            self._metrics.plans_from_cache += 1
            self._metrics.cache_hits += 1
        elif source == "llm":
            self._metrics.plans_from_llm += 1
            self._metrics.cache_misses += 1
        else:
            self._metrics.plans_from_pattern += 1
            self._metrics.cache_misses += 1
        
        # Update averages
        self._metrics.avg_plan_time_ms = (
            self._metrics.total_plan_time_ms / self._metrics.plans_generated
        )
        
        # Track step counts for Average
        current_avg = self._metrics.avg_steps_per_plan
        n = self._metrics.plans_generated
        self._metrics.avg_steps_per_plan = current_avg + (step_count - current_avg) / n
    
    def record_step_executed(
        self,
        step_id: str,
        success: bool,
        time_ms: int,
    ) -> None:
        """Record a step execution.
        
        Args:
            step_id: Step identifier.
            success: Whether step succeeded.
            time_ms: Execution time.
        """
        self._metrics.steps_executed += 1
        if success:
            self._metrics.steps_succeeded += 1
        else:
            self._metrics.steps_failed += 1
        
        self._metrics.total_step_time_ms += time_ms
        self._metrics.avg_step_time_ms = (
            self._metrics.total_step_time_ms / self._metrics.steps_executed
        )
    
    def record_parallel_batch(self, batch_size: int) -> None:
        """Record a parallel batch execution.
        
        Args:
            batch_size: Number of steps in batch.
        """
        self._metrics.parallel_batches += 1
        self._metrics.parallel_steps += batch_size
    
    def record_revision(self, category: str) -> None:
        """Record a plan revision.
        
        Args:
            category: Failure category that triggered revision.
        """
        self._metrics.revisions_total += 1
        self._metrics.revisions_by_category[category] = (
            self._metrics.revisions_by_category.get(category, 0) + 1
        )
    
    def record_qa_evaluation(self, accepted: bool) -> None:
        """Record a QA evaluation.
        
        Args:
            accepted: Whether patch was accepted.
        """
        self._metrics.qa_evaluations += 1
        if not accepted:
            self._metrics.qa_rejections += 1
    
    def start_plan_timer(self) -> None:
        """Start timing plan generation."""
        self._plan_start = time.monotonic()
    
    def stop_plan_timer(self) -> int:
        """Stop plan timer and return elapsed ms."""
        if self._plan_start:
            elapsed = int((time.monotonic() - self._plan_start) * 1000)
            self._plan_start = None
            return elapsed
        return 0
    
    def start_step_timer(self) -> None:
        """Start timing step execution."""
        self._step_start = time.monotonic()
    
    def stop_step_timer(self) -> int:
        """Stop step timer and return elapsed ms."""
        if self._step_start:
            elapsed = int((time.monotonic() - self._step_start) * 1000)
            self._step_start = None
            return elapsed
        return 0
    
    def get_metrics(self) -> PlannerMetrics:
        """Get current metrics."""
        return self._metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = PlannerMetrics()
        self._start_time = time.monotonic()
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string.
        """
        m = self._metrics
        lines = [
            "# HELP planner_plans_total Total plans generated",
            "# TYPE planner_plans_total counter",
            f'planner_plans_total{{source="cache"}} {m.plans_from_cache}',
            f'planner_plans_total{{source="llm"}} {m.plans_from_llm}',
            f'planner_plans_total{{source="pattern"}} {m.plans_from_pattern}',
            "",
            "# HELP planner_steps_total Total steps executed",
            "# TYPE planner_steps_total counter",
            f'planner_steps_total{{status="success"}} {m.steps_succeeded}',
            f'planner_steps_total{{status="failed"}} {m.steps_failed}',
            "",
            "# HELP planner_revisions_total Total revisions",
            "# TYPE planner_revisions_total counter",
            f"planner_revisions_total {m.revisions_total}",
            "",
            "# HELP planner_qa_evaluations_total Total QA evaluations",
            "# TYPE planner_qa_evaluations_total counter",
            f'planner_qa_evaluations_total{{status="accepted"}} {m.qa_evaluations - m.qa_rejections}',
            f'planner_qa_evaluations_total{{status="rejected"}} {m.qa_rejections}',
            "",
            "# HELP planner_parallel_batches_total Total parallel batches",
            "# TYPE planner_parallel_batches_total counter",
            f"planner_parallel_batches_total {m.parallel_batches}",
            "",
            "# HELP planner_cache_hit_rate Cache hit rate",
            "# TYPE planner_cache_hit_rate gauge",
            f"planner_cache_hit_rate {m.cache_hit_rate:.4f}",
            "",
            "# HELP planner_step_success_rate Step success rate",
            "# TYPE planner_step_success_rate gauge",
            f"planner_step_success_rate {m.step_success_rate:.4f}",
            "",
            "# HELP planner_avg_plan_time_ms Average plan generation time",
            "# TYPE planner_avg_plan_time_ms gauge",
            f"planner_avg_plan_time_ms {m.avg_plan_time_ms:.2f}",
            "",
            "# HELP planner_avg_step_time_ms Average step execution time",
            "# TYPE planner_avg_step_time_ms gauge",
            f"planner_avg_step_time_ms {m.avg_step_time_ms:.2f}",
        ]
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """Export metrics as JSON string."""
        import json
        return json.dumps(self._metrics.to_dict(), indent=2)


# Global metrics collector
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics() -> None:
    """Reset global metrics."""
    global _global_collector
    _global_collector = MetricsCollector()
