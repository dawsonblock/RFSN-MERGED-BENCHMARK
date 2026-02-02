"""Persistence for QA claim outcomes.
from __future__ import annotations

Stores claim-level outcomes for learning:
- Which claims tend to fail for given failure signatures
- Which evidence types correlate with rejections
- Negative memory for claim patterns
"""

import json
import logging
import os
import sqlite3

from .qa_types import ClaimType, QAAttempt, Verdict

logger = logging.getLogger(__name__)


class QAPersistence:
    """SQLite-backed storage for QA outcomes."""

    def __init__(self, db_path: str):
        """Initialize persistence.
        
        Args:
            db_path: Path to SQLite database.
        """
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claim_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                verdict_reason TEXT,
                evidence_types TEXT,
                failure_signature TEXT,
                diff_stats TEXT,
                created_ts INTEGER NOT NULL,
                UNIQUE(attempt_id, claim_id)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_claim_type_verdict
            ON claim_outcomes (claim_type, verdict)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_failure_signature
            ON claim_outcomes (failure_signature)
        """)

        # Negative memory table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS negative_claim_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_type TEXT NOT NULL,
                failure_signature TEXT NOT NULL,
                rejection_count INTEGER NOT NULL DEFAULT 1,
                last_rejection_ts INTEGER NOT NULL,
                evidence_types TEXT,
                UNIQUE(claim_type, failure_signature)
            )
        """)
        self.conn.commit()

    def record_attempt(
        self,
        attempt: QAAttempt,
        *,
        failure_signature: str = "",
        diff_stats: dict[str, int] | None = None,
        timestamp: int = 0,
    ) -> None:
        """Record a complete QA attempt.
        
        Args:
            attempt: The QA attempt.
            failure_signature: Normalized failure signature.
            diff_stats: Patch diff statistics.
            timestamp: Record timestamp.
        """
        import time
        timestamp = timestamp or int(time.time())
        diff_stats_json = json.dumps(diff_stats or {})

        for claim in attempt.claims:
            verdict = attempt.get_verdict(claim.id)
            if not verdict:
                continue

            evidence_types = [
                e.type.value for e in attempt.evidence
                if any(claim.type.value in str(e.data) for _ in [1])
            ]

            try:
                self.conn.execute("""
                    INSERT INTO claim_outcomes (
                        attempt_id, claim_id, claim_type, claim_text,
                        verdict, verdict_reason, evidence_types,
                        failure_signature, diff_stats, created_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(attempt_id, claim_id) DO UPDATE SET
                        verdict = excluded.verdict,
                        verdict_reason = excluded.verdict_reason
                """, (
                    attempt.attempt_id,
                    claim.id,
                    claim.type.value,
                    claim.text,
                    verdict.verdict.value,
                    verdict.reason,
                    json.dumps(evidence_types),
                    failure_signature,
                    diff_stats_json,
                    timestamp,
                ))

                # Update negative memory for rejections
                if verdict.verdict == Verdict.REJECT:
                    self._record_negative_memory(
                        claim.type,
                        failure_signature,
                        evidence_types,
                        timestamp,
                    )
            except sqlite3.Error as e:
                logger.warning("Failed to record claim outcome: %s", e)

        self.conn.commit()

    def _record_negative_memory(
        self,
        claim_type: ClaimType,
        failure_signature: str,
        evidence_types: list[str],
        timestamp: int,
    ) -> None:
        """Record negative memory for a rejected claim."""
        try:
            self.conn.execute("""
                INSERT INTO negative_claim_memory (
                    claim_type, failure_signature, rejection_count,
                    last_rejection_ts, evidence_types
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(claim_type, failure_signature) DO UPDATE SET
                    rejection_count = rejection_count + 1,
                    last_rejection_ts = excluded.last_rejection_ts
            """, (
                claim_type.value,
                failure_signature,
                timestamp,
                json.dumps(evidence_types),
            ))
        except sqlite3.Error as e:
            logger.warning("Failed to record negative memory: %s", e)

    def query_failure_patterns(
        self,
        failure_signature: str,
        *,
        min_rejections: int = 2,
    ) -> list[tuple[str, int]]:
        """Query claim types that frequently fail for a failure signature.
        
        Args:
            failure_signature: The failure signature to query.
            min_rejections: Minimum rejection count to include.
        
        Returns:
            List of (claim_type, rejection_count) tuples.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT claim_type, rejection_count
            FROM negative_claim_memory
            WHERE failure_signature = ? AND rejection_count >= ?
            ORDER BY rejection_count DESC
        """, (failure_signature, min_rejections))
        return cur.fetchall()

    def query_claim_success_rate(
        self,
        claim_type: ClaimType,
        *,
        limit: int = 100,
    ) -> dict[str, float]:
        """Query success rate for a claim type.
        
        Args:
            claim_type: The claim type to query.
            limit: Maximum records to consider.
        
        Returns:
            Dict with accept_rate, reject_rate, challenge_rate.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT verdict, COUNT(*) as cnt
            FROM claim_outcomes
            WHERE claim_type = ?
            GROUP BY verdict
            ORDER BY created_ts DESC
            LIMIT ?
        """, (claim_type.value, limit))

        rows = cur.fetchall()
        total = sum(r[1] for r in rows)
        if total == 0:
            return {"accept_rate": 0.0, "reject_rate": 0.0, "challenge_rate": 0.0}

        rates = {"accept_rate": 0.0, "reject_rate": 0.0, "challenge_rate": 0.0}
        for verdict, count in rows:
            if verdict == "ACCEPT":
                rates["accept_rate"] = count / total
            elif verdict == "REJECT":
                rates["reject_rate"] = count / total
            elif verdict == "CHALLENGE":
                rates["challenge_rate"] = count / total

        return rates

    def should_skip_claim(
        self,
        claim_type: ClaimType,
        failure_signature: str,
        *,
        threshold: int = 5,
    ) -> bool:
        """Check if a claim type should be skipped for a failure signature.
        
        Based on negative memory: if consistently rejected, maybe skip.
        
        Args:
            claim_type: Claim type to check.
            failure_signature: Current failure signature.
            threshold: Rejection count threshold.
        
        Returns:
            True if claim should be skipped.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT rejection_count
            FROM negative_claim_memory
            WHERE claim_type = ? AND failure_signature = ?
        """, (claim_type.value, failure_signature))
        row = cur.fetchone()
        return row is not None and row[0] >= threshold

    def close(self) -> None:
        """Close database connection."""
        try:
            self.conn.close()
        except Exception:
            pass


def create_qa_persistence(db_path: str) -> QAPersistence:
    """Factory function for QAPersistence."""
    return QAPersistence(db_path)
