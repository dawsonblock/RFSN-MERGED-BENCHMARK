"""Outcome Learning Module for RFSN Agent.

This module implements outcome-based learning that records successful and failed
patch patterns to improve future predictions. It uses a simple SQLite-backed
memory to store outcomes and compute success probabilities.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =============================================================================
# OUTCOME LEARNING CONFIGURATION
# =============================================================================

OUTCOME_DB_PATH = os.environ.get(
    "RFSN_OUTCOME_DB", 
    str(Path.home() / ".cache" / "rfsn" / "outcomes.db")
)


@dataclass
class PatternOutcome:
    """Represents an outcome for a specific pattern."""
    pattern_key: str
    success_count: int
    failure_count: int
    last_updated: float
    
    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.5  # Prior: 50% success rate
        return self.success_count / self.total_attempts


class OutcomeLearner:
    """Learns from patch outcomes to improve future predictions.
    
    This learner:
    1. Records successful and failed patches
    2. Extracts patterns from patches (file types, change patterns)
    3. Computes success probabilities for new patches
    """
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or OUTCOME_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        """Ensure database exists with required schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                pattern_key TEXT PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_updated REAL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                patch_hash TEXT,
                pattern_keys TEXT,
                success INTEGER,
                timestamp REAL,
                metadata TEXT
            )
        """)
        conn.commit()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn
    
    def record_outcome(
        self, 
        patch_diff: str, 
        success: bool, 
        task_id: str = "",
        metadata: dict | None = None
    ) -> None:
        """Record the outcome of a patch attempt.
        
        Args:
            patch_diff: The unified diff of the patch
            success: Whether the patch resolved the issue
            task_id: Identifier for the task
            metadata: Additional metadata to store
        """
        import time
        
        patterns = self._extract_patterns(patch_diff)
        patch_hash = hashlib.sha256(patch_diff.encode()).hexdigest()[:16]
        
        conn = self._get_conn()
        
        # Update pattern outcomes
        for pattern_key in patterns:
            if success:
                conn.execute("""
                    INSERT INTO outcomes (pattern_key, success_count, failure_count, last_updated, metadata)
                    VALUES (?, 1, 0, ?, ?)
                    ON CONFLICT(pattern_key) DO UPDATE SET
                        success_count = success_count + 1,
                        last_updated = ?
                """, (pattern_key, time.time(), json.dumps(metadata or {}), time.time()))
            else:
                conn.execute("""
                    INSERT INTO outcomes (pattern_key, success_count, failure_count, last_updated, metadata)
                    VALUES (?, 0, 1, ?, ?)
                    ON CONFLICT(pattern_key) DO UPDATE SET
                        failure_count = failure_count + 1,
                        last_updated = ?
                """, (pattern_key, time.time(), json.dumps(metadata or {}), time.time()))
        
        # Record patch history
        conn.execute("""
            INSERT INTO patch_history (task_id, patch_hash, pattern_keys, success, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (task_id, patch_hash, json.dumps(patterns), int(success), time.time(), json.dumps(metadata or {})))
        
        conn.commit()
    
    def predict_success(self, patch_diff: str) -> float:
        """Predict the success probability of a patch.
        
        Args:
            patch_diff: The unified diff of the patch
            
        Returns:
            Probability of success (0.0 to 1.0)
        """
        patterns = self._extract_patterns(patch_diff)
        if not patterns:
            return 0.5  # Prior: 50% success rate
        
        outcomes = self._get_pattern_outcomes(patterns)
        if not outcomes:
            return 0.5  # Prior: 50% success rate
        
        # Weighted average based on pattern frequency
        total_weight = 0.0
        weighted_success = 0.0
        
        for outcome in outcomes:
            weight = outcome.total_attempts  # More data = more weight
            weighted_success += outcome.success_rate * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_success / total_weight
    
    def get_similar_patches(self, patch_diff: str, limit: int = 5) -> list[dict]:
        """Find similar patches from history.
        
        Args:
            patch_diff: The unified diff to find similar patches for
            limit: Maximum number of results
            
        Returns:
            List of similar patch records with success/failure info
        """
        patterns = self._extract_patterns(patch_diff)
        if not patterns:
            return []
        
        conn = self._get_conn()
        
        # Find patches with overlapping patterns
        results = []
        for pattern in patterns[:3]:  # Limit to top 3 patterns
            cursor = conn.execute("""
                SELECT task_id, patch_hash, pattern_keys, success, timestamp
                FROM patch_history
                WHERE pattern_keys LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{pattern}%", limit))
            
            for row in cursor:
                results.append({
                    "task_id": row[0],
                    "patch_hash": row[1],
                    "patterns": json.loads(row[2]),
                    "success": bool(row[3]),
                    "timestamp": row[4],
                })
        
        # Deduplicate and sort by timestamp
        seen_hashes = set()
        unique_results = []
        for r in results:
            if r["patch_hash"] not in seen_hashes:
                seen_hashes.add(r["patch_hash"])
                unique_results.append(r)
        
        return sorted(unique_results, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def _extract_patterns(self, patch_diff: str) -> list[str]:
        """Extract learning patterns from a patch diff.
        
        Patterns include:
        - File extension (e.g., "ext:py")
        - Change type (e.g., "adds_import", "modifies_function")
        - Error patterns (e.g., "fixes_assertion")
        """
        patterns = []
        
        lines = patch_diff.split("\n")
        files_touched = set()
        has_import_change = False
        has_function_change = False
        has_class_change = False
        has_assertion_fix = False
        additions = 0
        deletions = 0
        
        for line in lines:
            # Extract file paths
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                filepath = line[6:] if line.startswith("--- a/") else line[6:]
                if filepath and filepath != "/dev/null":
                    ext = Path(filepath).suffix
                    if ext:
                        patterns.append(f"ext:{ext[1:]}")  # Remove leading dot
                    files_touched.add(filepath)
            
            # Count additions/deletions
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
                if "import " in line:
                    has_import_change = True
                if "def " in line:
                    has_function_change = True
                if "class " in line:
                    has_class_change = True
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
                if "assert" in line.lower():
                    has_assertion_fix = True
        
        # Add semantic patterns
        if has_import_change:
            patterns.append("changes:imports")
        if has_function_change:
            patterns.append("changes:function")
        if has_class_change:
            patterns.append("changes:class")
        if has_assertion_fix:
            patterns.append("fixes:assertion")
        
        # Size patterns
        if additions + deletions <= 5:
            patterns.append("size:tiny")
        elif additions + deletions <= 20:
            patterns.append("size:small")
        elif additions + deletions <= 50:
            patterns.append("size:medium")
        else:
            patterns.append("size:large")
        
        # File count
        patterns.append(f"files:{min(len(files_touched), 5)}")
        
        return list(set(patterns))
    
    def _get_pattern_outcomes(self, patterns: list[str]) -> list[PatternOutcome]:
        """Get outcomes for a list of patterns."""
        conn = self._get_conn()
        outcomes = []
        
        for pattern in patterns:
            cursor = conn.execute(
                "SELECT pattern_key, success_count, failure_count, last_updated FROM outcomes WHERE pattern_key = ?",
                (pattern,)
            )
            row = cursor.fetchone()
            if row:
                outcomes.append(PatternOutcome(
                    pattern_key=row[0],
                    success_count=row[1],
                    failure_count=row[2],
                    last_updated=row[3],
                ))
        
        return outcomes
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        conn = self._get_conn()
        
        cursor = conn.execute("SELECT COUNT(*) FROM patch_history")
        total_patches = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM patch_history WHERE success = 1")
        successful_patches = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(*) FROM outcomes")
        total_patterns = cursor.fetchone()[0]
        
        return {
            "total_patches": total_patches,
            "successful_patches": successful_patches,
            "success_rate": successful_patches / total_patches if total_patches > 0 else 0.0,
            "total_patterns": total_patterns,
        }


# Global instance for easy access
_learner: OutcomeLearner | None = None


def get_outcome_learner() -> OutcomeLearner:
    """Get the global outcome learner instance."""
    global _learner
    if _learner is None:
        _learner = OutcomeLearner()
    return _learner


def record_patch_outcome(patch_diff: str, success: bool, task_id: str = "") -> None:
    """Convenience function to record a patch outcome."""
    get_outcome_learner().record_outcome(patch_diff, success, task_id)


def predict_patch_success(patch_diff: str) -> float:
    """Convenience function to predict patch success probability."""
    return get_outcome_learner().predict_success(patch_diff)
