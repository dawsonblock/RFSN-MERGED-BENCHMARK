"""Upstream Learner package.

Contextual bandit for choosing planner/strategy/prompt_variant.
Persists to .rfsn_state/policy/upstream_policy.json.

This package is UPSTREAM ONLY - it never touches the gate,
executes commands, or applies patches directly.
"""

from __future__ import annotations

from .learner import UpstreamLearner, Decision
from .features import Context, featurize, parse_failure_signals, repo_fingerprint
from .policy_store import PolicyStore

__all__ = [
    "UpstreamLearner",
    "Decision",
    "Context",
    "featurize",
    "parse_failure_signals",
    "repo_fingerprint",
    "PolicyStore",
]
