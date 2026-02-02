from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Any


def _stable_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _diff_line_count(diff: str) -> int:
    if not diff:
        return 0
    return len(diff.splitlines())


def _outcome_value(outcome: str) -> float:
    o = (outcome or "").lower()
    if o == "success":
        return 1.0
    if o == "partial":
        return 0.5
    return 0.0


def score_action(
    *,
    outcome: str,
    exec_time_ms: int,
    command_count: int,
    diff_lines: int,
    regressions: int,
) -> float:
    base = 100.0 * _outcome_value(outcome)
    base -= float(max(0, command_count)) * 1.0
    base -= float(max(0, diff_lines)) * 0.02
    base -= float(max(0, regressions)) * 50.0
    return base


@dataclass(frozen=True)
class ContextSignature:
    failure_class: str
    repo_type: str
    language: str
    env: dict[str, Any]
    attempt_bucket: int
    failing_test_file: str | None
    sig_prefix: str | None
    stalled: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "repo_type": self.repo_type,
            "language": self.language,
            "env": self.env,
            "attempt_bucket": self.attempt_bucket,
            "failing_test_file": self.failing_test_file,
            "sig_prefix": self.sig_prefix,
            "stalled": self.stalled,
        }

    def canonical_json(self) -> str:
        return _stable_json(self.as_dict())

    def context_hash(self) -> str:
        return _sha256(self.canonical_json())

    def env_hash(self) -> str:
        return _sha256(_stable_json(self.env))


@dataclass(frozen=True)
class ActionPrior:
    action_type: str
    action_key: str
    action_json: dict[str, Any]
    weight: float
    success_rate: float
    mean_score: float
    n: int


def format_action_priors(priors: list[ActionPrior]) -> str:
    if not priors:
        return ""
    lines = []
    for p in priors:
        line = (
            f"- {p.action_type} {p.action_key}"
            f" | success_rate={p.success_rate:.2f}"
            f" | mean_score={p.mean_score:.1f}"
            f" | n={p.n}"
            f" | weight={p.weight:.2f}"
        )
        lines.append(line)
    return "\n".join(lines)


class ActionOutcomeStore:
    def __init__(
        self,
        db_path: str,
        *,
        half_life_days: int = 14,
        max_age_days: int = 90,
        max_rows: int = 20000,
    ):
        self.db_path = db_path
        self.half_life_days = max(1, int(half_life_days))
        self.max_age_days = max(1, int(max_age_days))
        self.max_rows = max(1000, int(max_rows))
        _ensure_parent_dir(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self._housekeeping()
        self._next_created_ts = self._compute_next_created_ts()

    def _compute_next_created_ts(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(created_ts) FROM action_outcomes")
        newest = cur.fetchone()[0]
        if newest is None:
            return 1
        return int(newest) + 1

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_hash TEXT NOT NULL UNIQUE,
                source_run_id TEXT NOT NULL,
                created_ts INTEGER NOT NULL,

                context_hash TEXT NOT NULL,
                context_json TEXT NOT NULL,
                failure_class TEXT NOT NULL,
                repo_type TEXT NOT NULL,
                language TEXT NOT NULL,
                env_hash TEXT NOT NULL,
                attempt_bucket INTEGER NOT NULL,
                failing_test_file TEXT,
                sig_prefix TEXT,
                stalled INTEGER NOT NULL,

                action_type TEXT NOT NULL,
                action_key TEXT NOT NULL,
                action_json TEXT NOT NULL,

                outcome TEXT NOT NULL,
                score REAL NOT NULL,
                confidence_weight REAL NOT NULL,

                exec_time_ms INTEGER NOT NULL,
                command_count INTEGER NOT NULL,
                diff_lines INTEGER NOT NULL,
                regressions INTEGER NOT NULL
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_outcomes_lookup "
            "ON action_outcomes (repo_type, failure_class, language, created_ts);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_outcomes_action "
            "ON action_outcomes (action_type, action_key);"
        )
        self.conn.commit()

    def _housekeeping(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(created_ts) FROM action_outcomes")
        newest = cur.fetchone()[0]
        if newest is None:
            return
        cutoff = int(newest) - int(self.max_age_days)
        self.conn.execute(
            "DELETE FROM action_outcomes WHERE created_ts < ?",
            (int(cutoff),),
        )
        self.conn.execute(
            "DELETE FROM action_outcomes WHERE id NOT IN ("
            "SELECT id FROM action_outcomes "
            "ORDER BY created_ts DESC, id DESC "
            "LIMIT ?"
            ")",
            (self.max_rows,),
        )
        self.conn.commit()

    def record(
        self,
        *,
        source_run_id: str,
        context: ContextSignature,
        action_type: str,
        action_key: str,
        action_json: dict[str, Any],
        outcome: str,
        score: float,
        confidence_weight: float,
        exec_time_ms: int,
        command_count: int,
        diff_lines: int,
        regressions: int,
        created_ts: int | None = None,
    ) -> None:
        if created_ts is None:
            created_ts = int(self._next_created_ts)
            self._next_created_ts += 1
        else:
            created_ts = int(created_ts)
            if created_ts >= int(self._next_created_ts):
                self._next_created_ts = int(created_ts) + 1
        event_hash = _sha256(
            _stable_json(
                {
                    "created_ts": int(created_ts),
                    "context_hash": context.context_hash(),
                    "action_type": action_type,
                    "action_key": action_key,
                    "source_run_id": source_run_id,
                }
            )
        )
        try:
            self.conn.execute(
                """
                INSERT INTO action_outcomes (
                    event_hash, source_run_id, created_ts,
                    context_hash, context_json, failure_class, repo_type, language, env_hash,
                    attempt_bucket, failing_test_file, sig_prefix, stalled,
                    action_type, action_key, action_json,
                    outcome, score, confidence_weight,
                    exec_time_ms, command_count, diff_lines, regressions
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    event_hash,
                    source_run_id,
                    created_ts,
                    context.context_hash(),
                    context.canonical_json(),
                    context.failure_class,
                    context.repo_type,
                    context.language,
                    context.env_hash(),
                    int(context.attempt_bucket),
                    context.failing_test_file,
                    context.sig_prefix,
                    1 if context.stalled else 0,
                    action_type,
                    action_key,
                    _stable_json(action_json),
                    (outcome or "").lower(),
                    float(score),
                    float(confidence_weight),
                    int(exec_time_ms),
                    int(command_count),
                    int(diff_lines),
                    int(regressions),
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            return

    def _decay(self, created_ts: int, now_ts: int) -> float:
        age_units = max(
            0.0,
            float(now_ts - int(created_ts)),
        )
        lam = math.log(2.0) / float(self.half_life_days)
        return math.exp(-lam * age_units)

    def query_action_priors(
        self,
        context: ContextSignature,
        *,
        top_k: int = 6,
        candidate_limit: int = 400,
        min_similarity: float = 0.25,
        now_ts: int | None = None,
    ) -> list[ActionPrior]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                action_type,
                action_key,
                action_json,
                outcome,
                score,
                confidence_weight,
                created_ts,
                env_hash,
                attempt_bucket,
                failing_test_file,
                sig_prefix,
                stalled
            FROM action_outcomes
            WHERE repo_type = ? AND failure_class = ? AND language = ?
            ORDER BY created_ts DESC, id DESC
            LIMIT ?
            """,
            (
                context.repo_type,
                context.failure_class,
                context.language,
                int(candidate_limit),
            ),
        )
        rows = cur.fetchall()
        if now_ts is None:
            newest = max((int(r[7]) for r in rows), default=0)
            now_ts = newest
        else:
            now_ts = int(now_ts)

        def sim(row: tuple[Any, ...]) -> float:
            env_hash = row[8]
            attempt_bucket = row[9]
            failing_test_file = row[10]
            sig_prefix = row[11]
            stalled = bool(row[12])
            s = 0.0
            s += 0.45 if env_hash == context.env_hash() else 0.0
            s += 0.20 if int(attempt_bucket) == int(context.attempt_bucket) else 0.0
            s += (
                0.15
                if (failing_test_file and failing_test_file == context.failing_test_file)
                else 0.0
            )
            s += (
                0.10
                if (sig_prefix and context.sig_prefix and sig_prefix == context.sig_prefix)
                else 0.0
            )
            s += 0.10 if stalled == context.stalled else 0.0
            return s

        agg: dict[str, dict[str, Any]] = {}
        for r in rows:
            action_type = r[1]
            action_key = r[2]
            action_json_s = r[3]
            outcome = r[4]
            score_v = r[5]
            conf_w = r[6]
            created_ts = r[7]
            similarity = sim(r)
            if similarity < float(min_similarity):
                continue
            decay = self._decay(int(created_ts), now_ts)
            w = float(conf_w) * float(similarity) * float(decay)
            if action_key not in agg:
                try:
                    action_json = json.loads(action_json_s)
                except Exception:
                    action_json = {"raw": str(action_json_s)}
                agg[action_key] = {
                    "action_type": str(action_type),
                    "action_key": str(action_key),
                    "action_json": action_json,
                    "w_sum": 0.0,
                    "succ_sum": 0.0,
                    "score_sum": 0.0,
                    "n": 0,
                }
            a = agg[action_key]
            a["w_sum"] += w
            a["succ_sum"] += w * _outcome_value(str(outcome))
            a["score_sum"] += w * float(score_v)
            a["n"] += 1

        priors: list[ActionPrior] = []
        for a in agg.values():
            if a["w_sum"] <= 0.0:
                continue
            success_rate = float(a["succ_sum"]) / float(a["w_sum"])
            mean_score = float(a["score_sum"]) / float(a["w_sum"])
            priors.append(
                ActionPrior(
                    action_type=a["action_type"],
                    action_key=a["action_key"],
                    action_json=a["action_json"],
                    weight=float(a["w_sum"]),
                    success_rate=success_rate,
                    mean_score=mean_score,
                    n=int(a["n"]),
                )
            )

        priors.sort(key=lambda p: (-p.weight, -p.success_rate, -p.mean_score, p.action_key))
        return priors[: int(top_k)]


def make_context_signature(
    *,
    failure_class: str,
    repo_type: str,
    language: str,
    env: dict[str, Any],
    attempt_count: int,
    failing_test_file: str | None,
    sig: str | None,
    stalled: bool,
) -> ContextSignature:
    attempt_bucket = max(0, min(int(attempt_count), 9))
    sig_prefix = (sig or "")[:12] if sig else None
    return ContextSignature(
        failure_class=str(failure_class or ""),
        repo_type=str(repo_type or ""),
        language=str(language or ""),
        env=dict(env or {}),
        attempt_bucket=attempt_bucket,
        failing_test_file=failing_test_file,
        sig_prefix=sig_prefix,
        stalled=bool(stalled),
    )


def make_action_key_for_tool(tool: str, args: dict[str, Any]) -> str:
    return _sha256(_stable_json({"tool": tool, "args": args}))


def make_action_key_for_patch(diff: str) -> str:
    return _sha256(diff or "")


def make_action_json_for_patch(diff: str, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "diff_hash": make_action_key_for_patch(diff),
        "diff_lines": _diff_line_count(diff),
        "tags": tags or [],
    }
