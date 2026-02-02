"""SimulationGate: human-like "mental rehearsal" without tool execution.

This module implements a **non-bypassable advisory layer** that can be placed
between proposal generation and the thalamic gate selection.

Hard constraints:
 - NO shell execution
 - NO tool execution
 - NO patch application
 - ONLY score adjustment (saliency/urgency/surprise)

The intent is to mimic how humans simulate only when needed:
 - small, coarse, fast
 - only a few top candidates
 - only when impact/uncertainty is high
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...event_bus import SimpleEventBus
from ...types import Candidate
from ..action_types import ActionPayload, CodingAction
from ..proposal_generators import ProposalContext


_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


@dataclass
class SimDecision:
    slot_id: str
    action: str
    simulate: bool
    impact_score: float
    uncertainty_score: float
    risk_flags: List[str]
    saliency_before: float
    saliency_after: float
    urgency_before: float
    urgency_after: float
    surprise_before: float
    surprise_after: float
    timestamp: float


class SimulationGate:
    """Advisory-only gate that adjusts candidate scores.

    This class never touches the filesystem and never runs any subprocess.
    It only reads candidate payloads, does cheap string heuristics, and
    adjusts scores to reduce catastrophic proposals and thrashing.
    """

    def __init__(self, event_bus: SimpleEventBus):
        self.event_bus = event_bus
        # simple cooldown cache to avoid re-evaluating identical diffs every cycle
        self._last_key: Optional[str] = None
        self._last_time: float = 0.0

    def adjust_candidates(self, candidates: List[Candidate], context: ProposalContext) -> None:
        """Adjust candidates in place.

        Strategy:
          1. Estimate impact (diff size / file sensitivity)
          2. Estimate uncertainty (how messy the state is)
          3. If impact*uncertainty is high, penalize risky candidates
             and optionally boost safer "inspect" actions.
        """

        # Quick cache: if nothing significant changed, don't churn scores.
        key = self._cache_key(candidates, context)
        now = time.time()
        if key == self._last_key and (now - self._last_time) < 0.25:
            return
        self._last_key, self._last_time = key, now

        decisions: List[SimDecision] = []

        # Only consider a few candidates (human-like short list).
        # Forced signals are handled elsewhere; here we only see normal candidates.
        top = sorted(candidates, key=lambda c: c.score(), reverse=True)[:3]
        top_ids = {c.slot_id for c in top}

        for c in candidates:
            payload = ActionPayload.from_bytes(c.content_payload)
            impact, flags = self._estimate_impact(payload)
            uncertainty = self._estimate_uncertainty(context, payload)

            simulate = (impact * uncertainty) >= 0.35 and c.slot_id in top_ids

            sal_before, urg_before, sur_before = c.saliency, c.urgency, c.surprise

            # Default: no change
            sal_after, urg_after, sur_after = sal_before, urg_before, sur_before

            if simulate:
                # Penalize risky high-impact actions, boost low-risk inspection.
                if payload.action == CodingAction.APPLY_PATCH:
                    # Reduce saliency proportionally to risk.
                    penalty = min(0.35, 0.15 + 0.5 * impact * uncertainty)
                    sal_after = max(0.0, sal_before * (1.0 - penalty))
                    # Increase "surprise" slightly to surface the risk as a signal.
                    sur_after = min(1.0, sur_before + 0.15)
                elif payload.action in {CodingAction.RUN_TESTS, CodingAction.RUN_FOCUSED_TESTS, CodingAction.READ_FILE, CodingAction.SEARCH_CODE}:
                    boost = min(0.25, 0.05 + 0.3 * uncertainty)
                    sal_after = min(1.0, sal_before * (1.0 + boost))
                # small urgency nudge if we're stuck
                if context.last_result is not None and not context.tests_passing:
                    urg_after = min(1.0, urg_before + 0.05)

            # Apply modifications in place
            c.saliency, c.urgency, c.surprise = sal_after, urg_after, sur_after

            decisions.append(SimDecision(
                slot_id=c.slot_id,
                action=payload.action.value,
                simulate=simulate,
                impact_score=impact,
                uncertainty_score=uncertainty,
                risk_flags=flags,
                saliency_before=sal_before,
                saliency_after=sal_after,
                urgency_before=urg_before,
                urgency_after=urg_after,
                surprise_before=sur_before,
                surprise_after=sur_after,
                timestamp=now,
            ))

        # Emit a compact audit event (no payload bytes)
        self.event_bus.emit("SIM_GATE", {
            "cycle_id": context.cycle_id,
            "decisions": [d.__dict__ for d in decisions],
            "timestamp": now,
        })

    def _estimate_uncertainty(self, context: ProposalContext, payload: ActionPayload) -> float:
        """Estimate uncertainty in [0,1]."""
        # Base uncertainty: failing tests count (more failures = more uncertain)
        n_fail = len(context.failing_tests or [])
        base = min(1.0, 0.2 + 0.12 * n_fail)

        # If last action failed, increase uncertainty slightly
        if context.last_result is not None and not context.tests_passing:
            base = min(1.0, base + 0.1)

        # If the proposal is an immediate patch without inspection, treat as higher uncertainty
        if payload.action == CodingAction.APPLY_PATCH and not (context.test_output or "").strip():
            base = min(1.0, base + 0.1)

        return base

    def _estimate_impact(self, payload: ActionPayload) -> Tuple[float, List[str]]:
        """Estimate impact in [0,1] and return risk flags."""
        flags: List[str] = []
        impact = 0.05

        diff = ""
        if payload.action == CodingAction.APPLY_PATCH:
            diff = str(payload.parameters.get("diff", ""))

            # size proxy
            lines = diff.count("\n") + 1 if diff else 0
            impact = min(1.0, 0.1 + lines / 4000.0)

            # sensitive file heuristic
            touched = self._diff_touched_files(diff)
            if any(self._is_sensitive_path(p) for p in touched):
                flags.append("sensitive_path")
                impact = min(1.0, impact + 0.25)

            if any(p.endswith(".github/workflows/") or p.startswith(".github/workflows/") for p in touched):
                flags.append("ci_workflow")
                impact = min(1.0, impact + 0.15)

            if len(touched) >= 6:
                flags.append("many_files")
                impact = min(1.0, impact + 0.15)

        return impact, flags

    def _diff_touched_files(self, diff: str) -> List[str]:
        return _DIFF_FILE_RE.findall(diff or "")

    def _is_sensitive_path(self, path: str) -> bool:
        sensitive_prefixes = (
            "rfsn_controller/executor_spine.py",
            "rfsn_controller/sandbox.py",
            "cgw_ssl_guard/thalamic_gate.py",
            "cgw_ssl_guard/cgw_state.py",
            "pyproject.toml",
            "requirements",
        )
        return any(path.startswith(p) for p in sensitive_prefixes)

    def _cache_key(self, candidates: List[Candidate], context: ProposalContext) -> str:
        h = hashlib.sha256()
        h.update(str(context.cycle_id).encode("utf-8"))
        h.update(str(len(context.failing_tests or [])).encode("utf-8"))
        # only hash the top few candidates to avoid churn
        for c in sorted(candidates, key=lambda c: c.score(), reverse=True)[:3]:
            h.update(c.slot_id.encode("utf-8"))
            h.update(str(c.saliency).encode("utf-8"))
        return h.hexdigest()[:16]
