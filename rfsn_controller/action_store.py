"""ActionOutcomeStore for learning from past actions.

Stores outcomes of actions to enable learning patterns
of what works and what doesn't for similar situations.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionOutcome:
    """Record of an action and its outcome."""
    
    action_type: str  # "patch", "test", "install", etc.
    context_hash: str  # Hash of relevant context
    input_summary: str  # Brief description of input
    success: bool
    outcome_details: dict[str, Any]
    timestamp: float
    
    # Optional learning data
    error_type: str | None = None
    recovery_action: str | None = None


@dataclass
class ActionOutcomeStore:
    """SQLite-based store for action outcomes.
    
    Enables:
    1. Learning from failures (what went wrong, how to recover)
    2. Predicting success probability for similar actions
    3. Suggesting alternative approaches based on history
    """
    
    db_path: str
    max_entries: int = 50000
    
    _conn: sqlite3.Connection | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        """Create database and tables."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS action_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT,
                context_hash TEXT,
                input_summary TEXT,
                success INTEGER,
                outcome_details TEXT,
                error_type TEXT,
                recovery_action TEXT,
                timestamp REAL,
                UNIQUE(action_type, context_hash, input_summary)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_action_type 
            ON action_outcomes(action_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_context 
            ON action_outcomes(context_hash)
        """)
        self._conn.commit()
    
    def _hash_context(self, context: dict[str, Any]) -> str:
        """Create hash from context dict."""
        # Include only stable keys for hashing
        stable = {
            k: v for k, v in context.items()
            if k in ("language", "test_cmd", "framework", "file_pattern")
        }
        key = json.dumps(stable, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def record(self, outcome: ActionOutcome) -> None:
        """Record an action outcome.
        
        Args:
            outcome: The action outcome to store.
        """
        if not self._conn:
            return
        
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO action_outcomes
                    (action_type, context_hash, input_summary, success, 
                     outcome_details, error_type, recovery_action, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.action_type,
                        outcome.context_hash,
                        outcome.input_summary[:500],
                        1 if outcome.success else 0,
                        json.dumps(outcome.outcome_details),
                        outcome.error_type,
                        outcome.recovery_action,
                        outcome.timestamp,
                    )
                )
                self._conn.commit()
                self._prune()
            except Exception:
                pass
    
    def record_from_controller(
        self,
        action_type: str,
        context: dict[str, Any],
        input_summary: str,
        success: bool,
        details: dict[str, Any] | None = None,
        error_type: str | None = None,
        recovery_action: str | None = None,
    ) -> None:
        """Convenience method to record from controller.
        
        Args:
            action_type: Type of action.
            context: Context dict with language, test_cmd, etc.
            input_summary: Brief description.
            success: Whether action succeeded.
            details: Additional outcome details.
            error_type: Classification of error if any.
            recovery_action: What was done to recover.
        """
        outcome = ActionOutcome(
            action_type=action_type,
            context_hash=self._hash_context(context),
            input_summary=input_summary,
            success=success,
            outcome_details=details or {},
            timestamp=time.time(),
            error_type=error_type,
            recovery_action=recovery_action,
        )
        self.record(outcome)
    
    def get_success_rate(
        self,
        action_type: str,
        context: dict[str, Any],
    ) -> tuple[float, int]:
        """Get historical success rate for action type in context.
        
        Args:
            action_type: Type of action.
            context: Context dict.
            
        Returns:
            (success_rate, sample_count) tuple.
        """
        if not self._conn:
            return 0.5, 0
        
        context_hash = self._hash_context(context)
        
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT success FROM action_outcomes
                WHERE action_type = ? AND context_hash = ?
                ORDER BY timestamp DESC LIMIT 100
                """,
                (action_type, context_hash)
            )
            
            rows = cursor.fetchall()
            if not rows:
                return 0.5, 0
            
            successes = sum(1 for (s,) in rows if s)
            return successes / len(rows), len(rows)
    
    def get_common_errors(
        self,
        action_type: str,
        context: dict[str, Any],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get most common error types for action in context.
        
        Args:
            action_type: Type of action.
            context: Context dict.
            limit: Max errors to return.
            
        Returns:
            List of {error_type, count, recovery_action} dicts.
        """
        if not self._conn:
            return []
        
        context_hash = self._hash_context(context)
        
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT error_type, COUNT(*) as cnt, recovery_action
                FROM action_outcomes
                WHERE action_type = ? AND context_hash = ? 
                  AND success = 0 AND error_type IS NOT NULL
                GROUP BY error_type
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (action_type, context_hash, limit)
            )
            
            return [
                {"error_type": et, "count": cnt, "recovery_action": ra}
                for et, cnt, ra in cursor.fetchall()
            ]
    
    def suggest_recovery(
        self,
        action_type: str,
        context: dict[str, Any],
        error_type: str,
    ) -> str | None:
        """Suggest recovery action based on history.
        
        Args:
            action_type: Type of action.
            context: Context dict.
            error_type: The error that occurred.
            
        Returns:
            Suggested recovery action or None.
        """
        if not self._conn:
            return None
        
        context_hash = self._hash_context(context)
        
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT recovery_action FROM action_outcomes
                WHERE action_type = ? AND context_hash = ? 
                  AND error_type = ? AND recovery_action IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (action_type, context_hash, error_type)
            )
            
            row = cursor.fetchone()
            return row[0] if row else None
    
    def _prune(self) -> None:
        """Remove old entries if over limit."""
        if not self._conn:
            return
        
        cursor = self._conn.execute("SELECT COUNT(*) FROM action_outcomes")
        count = cursor.fetchone()[0]
        
        if count > self.max_entries:
            delete_count = count - int(self.max_entries * 0.8)
            self._conn.execute(
                """
                DELETE FROM action_outcomes WHERE id IN (
                    SELECT id FROM action_outcomes 
                    ORDER BY timestamp ASC LIMIT ?
                )
                """,
                (delete_count,)
            )
            self._conn.commit()


# Global store instance
_store: ActionOutcomeStore | None = None
_store_lock = threading.Lock()


def get_action_store(db_path: str | None = None) -> ActionOutcomeStore:
    """Get the global action outcome store."""
    global _store
    with _store_lock:
        if _store is None:
            default_path = os.path.expanduser("~/.cache/rfsn/action_outcomes.db")
            _store = ActionOutcomeStore(db_path=db_path or default_path)
        return _store
