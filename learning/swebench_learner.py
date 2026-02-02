from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from .outcomes import Outcome, score_patch_quality
from .state_store import atomic_write_json, read_json
from .thompson_persist import PersistentThompsonBandit
from .epsgreedy_persist import PersistentStrategyBandit


@dataclass(frozen=True)
class Bucket:
    name: str


# Small but useful SWE-bench buckets (expand later)
BUCKETS = [
    Bucket("import_error"),
    Bucket("attribute_error"),
    Bucket("type_error"),
    Bucket("value_error"),
    Bucket("key_error"),
    Bucket("index_error"),
    Bucket("assertion_failed"),
    Bucket("snapshot_mismatch"),
    Bucket("api_signature_mismatch"),
    Bucket("dependency_version"),
    Bucket("timeout_or_hang"),
    Bucket("unknown"),
]


DEFAULT_TEMPLATES = {
    # Templates here are “plan skeletons” that upstream modules can pick,
    # not direct code changes.
    "minimal_local_fix": {"max_files": 2, "max_lines": 80, "style": "surgical"},
    "guard_and_validate": {"max_files": 2, "max_lines": 120, "style": "add-guards"},
    "api_compat_layer": {"max_files": 4, "max_lines": 200, "style": "compat"},
    "dependency_pin": {"max_files": 2, "max_lines": 60, "style": "config"},
    "regression_hunt": {"max_files": 6, "max_lines": 250, "style": "bisect-lite"},
}


def classify_bucket(test_output: str) -> str:
    t = (test_output or "").lower()

    if "modulenotfounderror" in t or "importerror" in t:
        return "import_error"
    if "attributeerror" in t:
        return "attribute_error"
    if "typeerror" in t or "unsupported operand" in t:
        return "type_error"
    if "valueerror" in t:
        return "value_error"
    if "keyerror" in t:
        return "key_error"
    if "indexerror" in t:
        return "index_error"
    if "assert" in t and ("failed" in t or "assertionerror" in t):
        return "assertion_failed"
    if "snapshot" in t and ("mismatch" in t or "changed" in t):
        return "snapshot_mismatch"
    if "takes" in t and "positional argument" in t:
        return "api_signature_mismatch"
    if "requires" in t and "version" in t:
        return "dependency_version"
    if "timeout" in t or "timed out" in t or "hang" in t:
        return "timeout_or_hang"
    return "unknown"


class SWEBenchLearner:
    """
    Upstream learner that:
      - picks planner via Thompson sampling
      - picks strategy via epsilon-greedy
      - maintains per-bucket and per-template stats
      - persists all state under .rfsn_state/
    """

    def __init__(self, state_dir: str = ".rfsn_state"):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)

        self.planner_bandit = PersistentThompsonBandit(
            path=os.path.join(self.state_dir, "bandits", "planner_thompson.json")
        )
        self.strategy_bandit = PersistentStrategyBandit(
            path=os.path.join(self.state_dir, "bandits", "strategy_epsgreedy.json"),
            epsilon=0.12,
        )

        self.bucket_stats_path = os.path.join(self.state_dir, "swebench", "bucket_stats.json")
        self.template_stats_path = os.path.join(self.state_dir, "swebench", "template_stats.json")
        self.episode_history_path = os.path.join(self.state_dir, "episodes", "episode_history.jsonl")

        self.bucket_stats = read_json(self.bucket_stats_path, default={"buckets": {}})
        self.template_stats = read_json(self.template_stats_path, default={"templates": DEFAULT_TEMPLATES, "stats": {}})

        # Ensure templates exist
        for k, v in DEFAULT_TEMPLATES.items():
            self.template_stats.setdefault("templates", {}).setdefault(k, v)
        self.template_stats.setdefault("stats", {})

    def choose_planner(self, options: Optional[List[str]] = None) -> str:
        return self.planner_bandit.choose(options or ["planner_v1"])

    def choose_strategy(self) -> str:
        # Strategies are higher-level than templates; they map to templates.
        return self.strategy_bandit.select()

    def strategy_to_template(self, strategy: str) -> str:
        # Keep this dumb and robust.
        mapping = {
            "default": "minimal_local_fix",
            "minimal": "minimal_local_fix",
            "guard": "guard_and_validate",
            "compat": "api_compat_layer",
            "deps": "dependency_pin",
            "regress": "regression_hunt",
        }
        return mapping.get(strategy, "minimal_local_fix")

    def record_episode(
        self,
        task_id: str,
        repo: str,
        bucket: str,
        planner: str,
        strategy: str,
        template: str,
        outcome: Outcome,
        patch_size: int,
        files_touched: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> float:
        r = score_patch_quality(outcome, patch_size=patch_size, files_touched=files_touched)

        # Update planner Thompson
        self.planner_bandit.update(planner, success=bool(outcome.passed), weight=max(1.0, abs(r)))
        self.planner_bandit.save()

        # Update strategy epsilon-greedy (running avg)
        self.strategy_bandit.update(strategy, reward=r)
        self.strategy_bandit.save()

        # Bucket stats
        b = self.bucket_stats.setdefault("buckets", {}).setdefault(bucket, {"n": 0, "success": 0, "reward_sum": 0.0})
        b["n"] += 1
        b["success"] += 1 if outcome.passed else 0
        b["reward_sum"] += float(r)

        # Template stats
        ts = self.template_stats.setdefault("stats", {}).setdefault(template, {"n": 0, "success": 0, "reward_sum": 0.0})
        ts["n"] += 1
        ts["success"] += 1 if outcome.passed else 0
        ts["reward_sum"] += float(r)

        # Persist stats
        atomic_write_json(self.bucket_stats_path, self.bucket_stats)
        atomic_write_json(self.template_stats_path, self.template_stats)

        # Append episode record
        os.makedirs(os.path.dirname(self.episode_history_path), exist_ok=True)
        rec = {
            "ts": time.time(),
            "task_id": task_id,
            "repo": repo,
            "bucket": bucket,
            "planner": planner,
            "strategy": strategy,
            "template": template,
            "passed": bool(outcome.passed),
            "test_delta": int(outcome.test_delta),
            "runtime": float(outcome.runtime),
            "patch_size": int(patch_size),
            "files_touched": int(files_touched),
            "reward": float(r),
            "error_message": outcome.error_message[:500] if outcome.error_message else "",
            "extra": extra or {},
        }
        with open(self.episode_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        return r

    def summarize(self) -> Dict[str, Any]:
        def agg_stats(d: Dict[str, Any]) -> List[Tuple[str, float, int, int]]:
            out = []
            for k, v in d.items():
                n = int(v.get("n", 0))
                s = int(v.get("success", 0))
                rs = float(v.get("reward_sum", 0.0))
                mean_r = (rs / n) if n else 0.0
                out.append((k, mean_r, s, n))
            out.sort(key=lambda t: (t[1], t[2]), reverse=True)
            return out

        buckets = agg_stats(self.bucket_stats.get("buckets", {}))
        templates = agg_stats(self.template_stats.get("stats", {}))

        return {
            "planner_bandit": self.planner_bandit.get_statistics(),
            "strategy_bandit": self.strategy_bandit.get_stats(),
            "bucket_rank": [{"bucket": b, "mean_reward": r, "success": s, "n": n} for (b, r, s, n) in buckets[:15]],
            "template_rank": [{"template": t, "mean_reward": r, "success": s, "n": n} for (t, r, s, n) in templates[:15]],
        }
