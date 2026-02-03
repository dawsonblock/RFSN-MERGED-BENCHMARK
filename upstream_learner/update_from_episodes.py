"""Update upstream policy from episode history.

Reads episode records from .rfsn_state/episodes/episode_history.jsonl
and updates the policy in .rfsn_state/policy/upstream_policy.json.

Usage:
    python -m upstream_learner.update_from_episodes
    python -m upstream_learner.update_from_episodes --episode_jsonl path/to/episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .features import Context, parse_failure_signals, repo_fingerprint
from .learner import Decision, UpstreamLearner


def score_reward(
    passed: bool,
    runtime_s: float,
    patch_size: int,
    files_touched: int,
    rc_k: int = 0,
    rc_top_wr: float = 0.5,
) -> float:
    """Compute reward from episode outcome.

    Pass dominates; penalties stop "huge patch spam".
    Repair-card shaping: rewards capitalizing on strong cards,
    penalizes wasting strong leads.

    Args:
        passed: Whether tests passed
        runtime_s: Runtime in seconds
        patch_size: Size of patch in bytes/chars
        files_touched: Number of files modified
        rc_k: Number of repair cards retrieved
        rc_top_wr: Win rate of top repair card (0-1)

    Returns:
        Reward value (higher is better)
    """
    r = 1.0 if passed else -1.0
    r -= 0.002 * float(patch_size)
    r -= 0.02 * float(files_touched)
    r -= 0.0005 * float(runtime_s)
    
    # Repair-card shaping: if strong cards were present,
    # reward passing more (capitalized on guidance)
    # penalize failing more (wasted strong lead)
    if rc_k > 0:
        strength = min(max(rc_top_wr, 0.0), 1.0)
        if passed:
            r += 0.15 * strength
        else:
            r -= 0.15 * strength
    
    return r


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Update upstream learner policy from episode history"
    )
    ap.add_argument(
        "--episode_jsonl",
        default=".rfsn_state/episodes/episode_history.jsonl",
        help="Path to episode history JSONL file",
    )
    ap.add_argument(
        "--repo_dir",
        default=".",
        help="Repository directory for fingerprinting",
    )
    args = ap.parse_args()

    learner = UpstreamLearner()
    fp = repo_fingerprint(args.repo_dir)

    if not os.path.exists(args.episode_jsonl):
        print(f"No episode history found: {args.episode_jsonl}")
        return 0

    n = 0
    with open(args.episode_jsonl, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

            # Use stored bucket if present; otherwise derive from error text
            bucket = ep.get("bucket", "unknown")

            # Extract repair-card stats from episode if logged
            rc_stats = (ep.get("extra", {}) or {}).get("repair_cards_stats", {}) or {}
            rc_k = int(rc_stats.get("rc_k", 0))
            rc_top_score = float(rc_stats.get("rc_top_score", 0.0))
            rc_top_wr = float(rc_stats.get("rc_top_wr", 0.5))

            # Build context from error_message (sanitized) + metadata
            sig = parse_failure_signals(ep.get("error_message", ""))
            ctx = Context(
                repo=str(ep.get("repo", "")),
                task_id=str(ep.get("task_id", "")),
                bucket=str(bucket),
                error_type=str(sig["error_type"]),
                top_module=str(sig["top_module"]),
                top_symbol=str(sig["top_symbol"]),
                test_hint=str(sig["test_hint"]),
                repo_fingerprint=fp,
                rc_k=rc_k,
                rc_top_score=rc_top_score,
                rc_top_wr=rc_top_wr,
            )

            # Decision keys: if not logged yet, fall back to ep fields
            extra = ep.get("extra", {}) or {}
            decision = Decision(
                planner=str(ep.get("planner", "planner_v1")),
                strategy=str(ep.get("strategy", "minimal")),
                prompt_variant=str(extra.get("prompt_variant", "p0_concise")),
            )

            reward = score_reward(
                passed=bool(ep.get("passed", False)),
                runtime_s=float(ep.get("runtime", 0.0)),
                patch_size=int(ep.get("patch_size", 0)),
                files_touched=int(ep.get("files_touched", 0)),
                rc_k=rc_k,
                rc_top_wr=rc_top_wr,
            )

            learner.update(ctx, decision, reward)
            n += 1

    print(f"Updated policy from {n} episodes -> {learner.store.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
