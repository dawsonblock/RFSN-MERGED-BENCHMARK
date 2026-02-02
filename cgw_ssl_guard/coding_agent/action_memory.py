"""CGW Action Outcome Memory Integration.

Integrates the rfsn_controller ActionOutcomeStore with the CGW coding agent.
Provides:
- Similarity-based proposal boosting from past successful actions
- Regression firewall to block patterns that consistently fail
- Memory stats for metrics dashboard
- Automatic recording of execution outcomes

This bridges the existing action_outcome_memory.py to work with the
CGW serial decision architecture.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Conditional import from rfsn_controller
try:
    from rfsn_controller.action_outcome_memory import (
        ActionOutcomeStore,
        ActionPrior,
        ContextSignature,
        make_action_key_for_patch,
        make_action_key_for_tool,
        make_action_json_for_patch,
        make_context_signature,
        score_action,
    )
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    ActionOutcomeStore = None
    ActionPrior = None
    ContextSignature = None


@dataclass
class CGWMemoryConfig:
    """Configuration for CGW action outcome memory."""
    
    # Database path (default: ~/.cgw/action_memory.db)
    db_path: Optional[str] = None
    
    # Half-life for temporal decay (days)
    half_life_days: int = 14
    
    # Maximum age to keep records (days)
    max_age_days: int = 90
    
    # Maximum rows in database
    max_rows: int = 20000
    
    # Minimum similarity for prior retrieval
    min_similarity: float = 0.25
    
    # Number of priors to retrieve
    top_k: int = 6
    
    # Regression firewall threshold (block if success_rate < this)
    regression_threshold: float = 0.2
    
    # Minimum observations for regression firewall
    min_observations: int = 3
    
    # Saliency boost multiplier for high-performing actions
    max_boost: float = 1.5
    
    def __post_init__(self):
        if self.db_path is None:
            home = os.path.expanduser("~")
            self.db_path = os.path.join(home, ".cgw", "action_memory.db")


class CGWActionMemory:
    """CGW-integrated action outcome memory.
    
    This class wraps the rfsn_controller ActionOutcomeStore and provides
    CGW-specific functionality:
    
    1. Similarity boosting: Query similar past contexts and boost proposals
       that worked well in those contexts.
    
    2. Regression firewall: Block action patterns that consistently fail
       to prevent wasting cycles on known-bad approaches.
    
    3. Outcome recording: Automatically record execution outcomes for
       future learning.
    
    Usage:
        memory = CGWActionMemory()
        
        # Before gate selection, get priors
        priors = memory.get_action_priors(context_dict)
        for prior in priors:
            boost_proposal(prior.action_type, prior.weight)
        
        # Check regression firewall
        if memory.is_blocked(action_type, action_key, context_dict):
            skip_action()
        
        # After execution, record outcome
        memory.record_outcome(
            action_type="APPLY_PATCH",
            action_key=patch_hash,
            outcome="success",
            context_dict=context_dict,
            exec_time_ms=1500,
        )
    """
    
    def __init__(
        self,
        config: Optional[CGWMemoryConfig] = None,
    ):
        self.config = config or CGWMemoryConfig()
        self._store: Optional[Any] = None
        self._session_id: str = ""
        self._current_run_id: str = ""
        
        if MEMORY_AVAILABLE:
            self._init_store()
        else:
            logger.warning(
                "action_outcome_memory not available. "
                "Memory features will be disabled."
            )
    
    def _init_store(self) -> None:
        """Initialize the underlying ActionOutcomeStore."""
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._store = ActionOutcomeStore(
            db_path=self.config.db_path,
            half_life_days=self.config.half_life_days,
            max_age_days=self.config.max_age_days,
            max_rows=self.config.max_rows,
        )
    
    def set_session(self, session_id: str, run_id: str = "") -> None:
        """Set the current session and run ID."""
        self._session_id = session_id
        self._current_run_id = run_id or f"{session_id}_{int(time.time())}"
    
    def build_context(
        self,
        *,
        failure_class: str = "unknown",
        repo_type: str = "unknown",
        language: str = "python",
        env: Optional[Dict[str, Any]] = None,
        attempt_count: int = 0,
        failing_test_file: Optional[str] = None,
        signature: Optional[str] = None,
        stalled: bool = False,
    ) -> Optional[Any]:
        """Build a ContextSignature for memory queries.
        
        Args:
            failure_class: Type of failure (e.g., "assertion", "import", "syntax")
            repo_type: Repository type (e.g., "python", "node", "java")
            language: Primary language
            env: Environment context (test framework, etc.)
            attempt_count: Number of attempts so far
            failing_test_file: Path to failing test
            signature: Error signature prefix
            stalled: Whether progress has stalled
            
        Returns:
            ContextSignature or None if memory not available.
        """
        if not MEMORY_AVAILABLE:
            return None
        
        return make_context_signature(
            failure_class=failure_class,
            repo_type=repo_type,
            language=language,
            env=env or {},
            attempt_count=attempt_count,
            failing_test_file=failing_test_file,
            sig=signature,
            stalled=stalled,
        )
    
    def get_action_priors(
        self,
        context: Optional[Any] = None,
        **context_kwargs,
    ) -> List[Any]:
        """Get action priors from memory.
        
        Args:
            context: Pre-built ContextSignature
            **context_kwargs: Arguments to build_context if context not provided
            
        Returns:
            List of ActionPrior objects sorted by weight.
        """
        if self._store is None:
            return []
        
        if context is None:
            context = self.build_context(**context_kwargs)
        
        if context is None:
            return []
        
        try:
            return self._store.query_action_priors(
                context,
                top_k=self.config.top_k,
                min_similarity=self.config.min_similarity,
            )
        except Exception as e:
            logger.warning(f"Failed to query priors: {e}")
            return []
    
    def get_saliency_boost(
        self,
        action_type: str,
        action_key: Optional[str] = None,
        context: Optional[Any] = None,
        **context_kwargs,
    ) -> float:
        """Get saliency boost for an action based on memory.
        
        Args:
            action_type: The action type to check
            action_key: Optional specific action key
            context: Pre-built ContextSignature
            **context_kwargs: Arguments for build_context
            
        Returns:
            Saliency multiplier (1.0 = no change, >1 = boost, <1 = penalize)
        """
        priors = self.get_action_priors(context, **context_kwargs)
        
        if not priors:
            return 1.0
        
        # Find matching priors
        for prior in priors:
            if prior.action_type == action_type:
                if action_key is None or prior.action_key == action_key:
                    # Boost based on success rate and weight
                    boost = 1.0 + (prior.success_rate * prior.weight * 
                                   (self.config.max_boost - 1.0))
                    return min(self.config.max_boost, boost)
        
        return 1.0
    
    def is_blocked(
        self,
        action_type: str,
        action_key: str,
        context: Optional[Any] = None,
        **context_kwargs,
    ) -> bool:
        """Check if an action is blocked by the regression firewall.
        
        An action is blocked if:
        1. We have enough observations (min_observations)
        2. The success rate is below the threshold
        
        Args:
            action_type: The action type to check
            action_key: The specific action key
            context: Pre-built ContextSignature
            **context_kwargs: Arguments for build_context
            
        Returns:
            True if action should be blocked.
        """
        priors = self.get_action_priors(context, **context_kwargs)
        
        for prior in priors:
            if prior.action_type == action_type and prior.action_key == action_key:
                if (prior.n >= self.config.min_observations and 
                    prior.success_rate < self.config.regression_threshold):
                    logger.info(
                        f"Regression firewall: blocking {action_type}:{action_key[:8]} "
                        f"(success_rate={prior.success_rate:.2f}, n={prior.n})"
                    )
                    return True
        
        return False
    
    def get_blocked_actions(
        self,
        context: Optional[Any] = None,
        **context_kwargs,
    ) -> List[Dict[str, Any]]:
        """Get all blocked actions for a context.
        
        Returns:
            List of dicts with blocked action info.
        """
        priors = self.get_action_priors(context, **context_kwargs)
        
        blocked = []
        for prior in priors:
            if (prior.n >= self.config.min_observations and 
                prior.success_rate < self.config.regression_threshold):
                blocked.append({
                    "action_type": prior.action_type,
                    "action_key": prior.action_key,
                    "success_rate": prior.success_rate,
                    "n": prior.n,
                })
        
        return blocked
    
    def record_outcome(
        self,
        *,
        action_type: str,
        action_key: str,
        outcome: str,
        context: Optional[Any] = None,
        action_json: Optional[Dict[str, Any]] = None,
        exec_time_ms: int = 0,
        command_count: int = 0,
        diff_lines: int = 0,
        regressions: int = 0,
        confidence_weight: float = 1.0,
        **context_kwargs,
    ) -> bool:
        """Record an action outcome to memory.
        
        Args:
            action_type: Type of action executed
            action_key: Unique key for the action
            outcome: "success", "partial", or "failure"
            context: Pre-built ContextSignature
            action_json: Additional action metadata
            exec_time_ms: Execution time in milliseconds
            command_count: Number of commands executed
            diff_lines: Number of diff lines
            regressions: Number of regressions caused
            confidence_weight: Weight for this observation
            **context_kwargs: Arguments for build_context
            
        Returns:
            True if recorded successfully.
        """
        if self._store is None:
            return False
        
        if context is None:
            context = self.build_context(**context_kwargs)
        
        if context is None:
            return False
        
        try:
            score = score_action(
                outcome=outcome,
                exec_time_ms=exec_time_ms,
                command_count=command_count,
                diff_lines=diff_lines,
                regressions=regressions,
            )
            
            self._store.record(
                source_run_id=self._current_run_id or "cgw",
                context=context,
                action_type=action_type,
                action_key=action_key,
                action_json=action_json or {},
                outcome=outcome,
                score=score,
                confidence_weight=confidence_weight,
                exec_time_ms=exec_time_ms,
                command_count=command_count,
                diff_lines=diff_lines,
                regressions=regressions,
            )
            
            logger.debug(
                f"Recorded outcome: {action_type} {outcome} "
                f"(score={score:.1f})"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to record outcome: {e}")
            return False
    
    def record_patch_outcome(
        self,
        *,
        diff: str,
        outcome: str,
        context: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> bool:
        """Convenience method to record a patch outcome.
        
        Args:
            diff: The diff/patch content
            outcome: "success", "partial", or "failure"
            context: Pre-built ContextSignature
            tags: Optional tags for the patch
            **kwargs: Additional arguments for record_outcome
        """
        if not MEMORY_AVAILABLE:
            return False
        
        action_key = make_action_key_for_patch(diff)
        action_json = make_action_json_for_patch(diff, tags)
        
        return self.record_outcome(
            action_type="APPLY_PATCH",
            action_key=action_key,
            outcome=outcome,
            context=context,
            action_json=action_json,
            diff_lines=len(diff.splitlines()) if diff else 0,
            **kwargs,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self._store is None:
            return {"available": False}
        
        try:
            cursor = self._store.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM action_outcomes")
            total_records = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT action_type, COUNT(*) FROM action_outcomes "
                "GROUP BY action_type"
            )
            by_type = dict(cursor.fetchall())
            
            cursor.execute(
                "SELECT outcome, COUNT(*) FROM action_outcomes "
                "GROUP BY outcome"
            )
            by_outcome = dict(cursor.fetchall())
            
            return {
                "available": True,
                "total_records": total_records,
                "by_action_type": by_type,
                "by_outcome": by_outcome,
                "db_path": self.config.db_path,
            }
        except Exception as e:
            return {"available": True, "error": str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        stats = self.get_stats()
        lines = [
            "# HELP cgw_memory_records_total Total records in memory",
            "# TYPE cgw_memory_records_total gauge",
        ]
        
        if stats.get("available"):
            lines.append(
                f'cgw_memory_records_total {stats.get("total_records", 0)}'
            )
            
            if "by_action_type" in stats:
                lines.extend([
                    "",
                    "# HELP cgw_memory_records_by_type Records by action type",
                    "# TYPE cgw_memory_records_by_type gauge",
                ])
                for action_type, count in stats["by_action_type"].items():
                    lines.append(
                        f'cgw_memory_records_by_type{{type="{action_type}"}} {count}'
                    )
        
        return "\n".join(lines)
    
    def close(self) -> None:
        """Close the underlying store."""
        if self._store is not None:
            self._store.close()
            self._store = None


# === Singleton Access ===

_memory_instance: Optional[CGWActionMemory] = None


def get_action_memory(
    db_path: Optional[str] = None,
) -> CGWActionMemory:
    """Get or create the global action memory instance."""
    global _memory_instance
    
    if _memory_instance is None:
        config = CGWMemoryConfig()
        if db_path:
            config.db_path = db_path
        _memory_instance = CGWActionMemory(config=config)
    
    return _memory_instance


def reset_action_memory() -> None:
    """Reset the global memory instance (for testing)."""
    global _memory_instance
    if _memory_instance:
        _memory_instance.close()
    _memory_instance = None


# === Executor Integration ===

class MemoryExecutorMixin:
    """Mixin to add memory recording to BlockingExecutor.
    
    Usage:
        class MyExecutor(BlockingExecutor, MemoryExecutorMixin):
            def execute(self, payload):
                result = super().execute(payload)
                self.record_to_memory(payload, result)
                return result
    """
    
    _memory: Optional[CGWActionMemory] = None
    _context: Optional[Any] = None
    
    def set_memory(self, memory: CGWActionMemory) -> None:
        """Set the memory instance."""
        self._memory = memory
    
    def set_context(self, context: Any) -> None:
        """Set the current context for memory queries."""
        self._context = context
    
    def record_to_memory(
        self,
        payload: Any,
        result: Any,
    ) -> None:
        """Record execution outcome to memory."""
        if self._memory is None:
            return
        
        action_type = payload.action.name if hasattr(payload, 'action') else "UNKNOWN"
        action_key = str(hash(str(payload)))
        
        success = getattr(result, 'success', False)
        outcome = "success" if success else "failure"
        
        exec_time_ms = int(getattr(result, 'execution_time', 0) * 1000)
        
        self._memory.record_outcome(
            action_type=action_type,
            action_key=action_key,
            outcome=outcome,
            context=self._context,
            exec_time_ms=exec_time_ms,
        )
