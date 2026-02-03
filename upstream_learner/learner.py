"""Upstream Learner - Contextual bandit for planner/strategy/prompt selection.

This learner is UPSTREAM ONLY:
- Never touches the gate
- Never executes commands
- Never applies patches directly
- Only emits advisory decisions

Authority boundary is strictly maintained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import Context, featurize
from .linucb import LinUCBArm
from .policy_store import PolicyStore


# Default arms for each decision head
DEFAULT_ARMS: dict[str, list[str]] = {
    "planner": [
        "planner_v1",
        "planner_traceback_first",
        "planner_api_compat",
        "planner_regression_hunt",
    ],
    "strategy": ["minimal", "guard", "compat", "deps", "regress"],
    "prompt": [
        "p0_concise",
        "p1_traceback",
        "p2_rag_patch",
        "p3_multi_plan",
        "v_traceback_local",
        "v_api_compat_shim",
        "v_multi_plan_select",
    ],
}

# Joint planner×prompt arms (cartesian product of key combinations)
# This learns which planner+prompt pairs work best together
JOINT_ARMS: list[str] = [
    # planner_v1 combinations
    "planner_v1|p0_concise",
    "planner_v1|p1_traceback",
    "planner_v1|p2_rag_patch",
    # traceback_first works best with traceback prompts
    "planner_traceback_first|v_traceback_local",
    "planner_traceback_first|p1_traceback",
    "planner_traceback_first|p0_concise",
    # api_compat works best with compat prompts
    "planner_api_compat|v_api_compat_shim",
    "planner_api_compat|p0_concise",
    "planner_api_compat|p2_rag_patch",
    # regression_hunt prefers multi-plan selection
    "planner_regression_hunt|v_multi_plan_select",
    "planner_regression_hunt|p3_multi_plan",
    "planner_regression_hunt|p0_concise",
]


@dataclass
class Decision:
    """Decision tuple from upstream learner."""

    planner: str
    strategy: str
    prompt_variant: str


class UpstreamLearner:
    """Contextual bandit for choosing planner/strategy/prompt_variant.

    Persists to one JSON policy file.

    Usage:
        learner = UpstreamLearner()
        decision = learner.decide(ctx)
        # ... run episode ...
        learner.update(ctx, decision, reward)
    """

    def __init__(self, store: PolicyStore | None = None, d: int = 15) -> None:
        """Initialize learner.

        Args:
            store: Policy storage backend (defaults to file-based)
            d: Feature dimension (default 15, includes repair-card features)
        """
        self.store = store or PolicyStore()
        self.d = d
        self.policy = self.store.load()
        self.policy.setdefault("bandits", {})
        self._ensure_bandits()

    def _ensure_bandits(self) -> None:
        """Ensure all default arms exist in policy with correct dimensions."""
        b = self.policy["bandits"]
        # Independent heads
        for head, arms in DEFAULT_ARMS.items():
            hb = b.setdefault(head, {"arms": {}})
            for a in arms:
                if a not in hb["arms"]:
                    hb["arms"][a] = LinUCBArm(d=self.d).to_dict()
                else:
                    # Check dimension and migrate if needed
                    stored = hb["arms"][a]
                    if stored.get("d", 12) != self.d:
                        # Reinitialize with correct dimension
                        hb["arms"][a] = LinUCBArm(d=self.d).to_dict()
        # Joint planner+prompt head
        jb = b.setdefault("joint", {"arms": {}})
        for ja in JOINT_ARMS:
            if ja not in jb["arms"]:
                jb["arms"][ja] = LinUCBArm(d=self.d).to_dict()
            else:
                stored = jb["arms"][ja]
                if stored.get("d", 12) != self.d:
                    jb["arms"][ja] = LinUCBArm(d=self.d).to_dict()

    def save(self) -> None:
        """Persist policy to disk."""
        self.store.save(self.policy)

    def _pick(self, head: str, x: list[float]) -> str:
        """Pick best arm for decision head given context."""
        hb = self.policy["bandits"][head]["arms"]
        best = None
        best_score = -1e18
        for name, arm_obj in hb.items():
            arm = LinUCBArm.from_dict(arm_obj)
            s = arm.score(x)
            if s > best_score:
                best_score = s
                best = name
        return best or list(hb.keys())[0]

    def decide(self, ctx: Context) -> Decision:
        """Make decision for given context using independent heads.

        Args:
            ctx: Context with task/failure metadata

        Returns:
            Decision with planner, strategy, prompt_variant
        """
        x = featurize(ctx)
        planner = self._pick("planner", x)
        strategy = self._pick("strategy", x)
        prompt = self._pick("prompt", x)
        return Decision(planner=planner, strategy=strategy, prompt_variant=prompt)

    def decide_joint(self, ctx: Context) -> Decision:
        """Make decision using joint planner+prompt arm.

        This learns which planner+prompt pairs work best together,
        avoiding the problem where they 'fight each other'.

        Args:
            ctx: Context with task/failure metadata

        Returns:
            Decision with planner, strategy, prompt_variant from joint selection
        """
        x = featurize(ctx)
        
        # Pick best joint arm
        jb = self.policy["bandits"].get("joint", {}).get("arms", {})
        if not jb:
            # Fallback to independent heads if no joint arms
            return self.decide(ctx)
        
        best_joint = None
        best_score = -1e18
        for name, arm_obj in jb.items():
            arm = LinUCBArm.from_dict(arm_obj)
            s = arm.score(x)
            if s > best_score:
                best_score = s
                best_joint = name
        
        if best_joint and "|" in best_joint:
            planner, prompt = best_joint.split("|", 1)
        else:
            # Fallback
            planner = self._pick("planner", x)
            prompt = self._pick("prompt", x)
        
        # Strategy is still independent
        strategy = self._pick("strategy", x)
        
        return Decision(planner=planner, strategy=strategy, prompt_variant=prompt)

    def update(self, ctx: Context, decision: Decision, reward: float) -> None:
        """Update learner with reward observation.

        Args:
            ctx: Context used for decision
            decision: Decision that was made
            reward: Observed reward (positive = good outcome)
        """
        x = featurize(ctx)
        
        # Update independent heads
        for head, chosen in [
            ("planner", decision.planner),
            ("strategy", decision.strategy),
            ("prompt", decision.prompt_variant),
        ]:
            hb = self.policy["bandits"][head]["arms"]
            if chosen not in hb:
                # Unknown arm - skip update
                continue
            arm = LinUCBArm.from_dict(hb[chosen])
            arm.update(x, reward)
            hb[chosen] = arm.to_dict()
        
        # Update joint arm (planner|prompt)
        joint_key = f"{decision.planner}|{decision.prompt_variant}"
        jb = self.policy["bandits"].get("joint", {}).get("arms", {})
        if joint_key in jb:
            arm = LinUCBArm.from_dict(jb[joint_key])
            arm.update(x, reward)
            jb[joint_key] = arm.to_dict()
        
        self.save()

    def get_arm_stats(self, head: str) -> dict[str, dict[str, Any]]:
        """Get statistics for all arms in a decision head.

        Args:
            head: Decision head (planner, strategy, prompt)

        Returns:
            Dict mapping arm name to stats (count, estimated_value)
        """
        hb = self.policy["bandits"].get(head, {}).get("arms", {})
        stats = {}
        for name, arm_obj in hb.items():
            arm = LinUCBArm.from_dict(arm_obj)
            # Estimate value as theta dot with bias feature
            x_bias = [0.0] * (self.d - 1) + [1.0]
            try:
                val = arm.score(x_bias)
            except Exception:
                val = 0.0
            stats[name] = {
                "estimated_value": val,
                "d": arm.d,
                "alpha": arm.alpha,
            }
        return stats
