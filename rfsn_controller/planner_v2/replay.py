"""Replay - Deterministic plan replay for debugging.

Replays a plan run with stored prompts/responses and verifies
that the controller produces the same diffs and outcomes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .artifact_log import PlanArtifact, StepArtifact
from .schema import ControllerOutcome, ControllerTaskSpec, Plan


@dataclass
class StepDivergence:
    """Describes a divergence between expected and actual step outcome."""
    
    step_id: str
    field: str
    expected: Any
    actual: Any
    
    def __str__(self) -> str:
        return f"Step {self.step_id}: {self.field} expected '{self.expected}' got '{self.actual}'"


@dataclass
class ReplayResult:
    """Result of a plan replay."""
    
    artifact_id: str
    plan_id: str
    success: bool
    divergences: list[StepDivergence] = field(default_factory=list)
    steps_replayed: int = 0
    steps_matched: int = 0
    error: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "plan_id": self.plan_id,
            "success": self.success,
            "divergences": [str(d) for d in self.divergences],
            "steps_replayed": self.steps_replayed,
            "steps_matched": self.steps_matched,
            "error": self.error,
        }


class PlanReplay:
    """Replays a plan run for verification and debugging."""
    
    def __init__(self, artifact: PlanArtifact):
        """Initialize with a plan artifact.
        
        Args:
            artifact: The plan artifact to replay.
        """
        self.artifact = artifact
        self.plan = Plan.from_json(artifact.plan_json)
    
    def replay(
        self,
        controller_fn: Callable[[ControllerTaskSpec], ControllerOutcome],
        dry_run: bool = False,
    ) -> ReplayResult:
        """Replay plan execution and verify outcomes match.
        
        Args:
            controller_fn: Function that executes a task spec and returns outcome.
            dry_run: If True, don't actually execute, just verify artifact structure.
            
        Returns:
            ReplayResult with divergence report.
        """
        divergences = []
        steps_replayed = 0
        steps_matched = 0
        
        try:
            for step_artifact in self.artifact.step_artifacts:
                expected_spec = ControllerTaskSpec.from_dict(
                    json.loads(step_artifact.task_spec_json)
                )
                expected_outcome = ControllerOutcome.from_dict(
                    json.loads(step_artifact.outcome_json)
                )
                
                if dry_run:
                    # Just validate artifact structure
                    steps_replayed += 1
                    steps_matched += 1
                    continue
                
                # Execute step
                actual_outcome = controller_fn(expected_spec)
                steps_replayed += 1
                
                # Compare outcomes
                step_divergences = self.compare_outcome(
                    expected_outcome, actual_outcome
                )
                
                if step_divergences:
                    divergences.extend(step_divergences)
                else:
                    steps_matched += 1
            
            return ReplayResult(
                artifact_id=f"{self.artifact.plan_id}",
                plan_id=self.artifact.plan_id,
                success=len(divergences) == 0,
                divergences=divergences,
                steps_replayed=steps_replayed,
                steps_matched=steps_matched,
            )
            
        except Exception as e:
            return ReplayResult(
                artifact_id=f"{self.artifact.plan_id}",
                plan_id=self.artifact.plan_id,
                success=False,
                divergences=divergences,
                steps_replayed=steps_replayed,
                steps_matched=steps_matched,
                error=str(e),
            )
    
    def compare_outcome(
        self,
        expected: ControllerOutcome,
        actual: ControllerOutcome,
    ) -> list[StepDivergence]:
        """Compare expected and actual outcomes.
        
        Args:
            expected: Expected outcome from artifact.
            actual: Actual outcome from replay.
            
        Returns:
            List of divergences found.
        """
        divergences = []
        
        # Compare critical fields
        if expected.success != actual.success:
            divergences.append(StepDivergence(
                step_id=expected.step_id,
                field="success",
                expected=expected.success,
                actual=actual.success,
            ))
        
        if expected.patch_applied != actual.patch_applied:
            divergences.append(StepDivergence(
                step_id=expected.step_id,
                field="patch_applied",
                expected=expected.patch_applied,
                actual=actual.patch_applied,
            ))
        
        if expected.tests_passed != actual.tests_passed:
            divergences.append(StepDivergence(
                step_id=expected.step_id,
                field="tests_passed",
                expected=expected.tests_passed,
                actual=actual.tests_passed,
            ))
        
        return divergences
    
    def get_step_trace(self, step_id: str) -> StepArtifact | None:
        """Get the artifact for a specific step.
        
        Args:
            step_id: The step ID to look up.
            
        Returns:
            StepArtifact or None.
        """
        for step in self.artifact.step_artifacts:
            if step.step_id == step_id:
                return step
        return None
    
    def get_failure_explanation(self, step_id: str) -> str:
        """Get human-readable explanation for step failure.
        
        Args:
            step_id: The step ID.
            
        Returns:
            Explanation string.
        """
        step = self.get_step_trace(step_id)
        if not step:
            return f"Step {step_id} not found in artifact"
        
        outcome = ControllerOutcome.from_dict(json.loads(step.outcome_json))
        spec = ControllerTaskSpec.from_dict(json.loads(step.task_spec_json))
        
        lines = [
            f"Step: {step_id}",
            f"Intent: {spec.intent}",
            f"Success: {outcome.success}",
        ]
        
        if not outcome.success:
            lines.append(f"Error: {outcome.error_message or 'Unknown'}")
            if outcome.failure_evidence:
                ev = outcome.failure_evidence
                lines.append(f"Category: {ev.category.value}")
                if ev.top_failing_tests:
                    lines.append(f"Failing tests: {', '.join(ev.top_failing_tests[:3])}")
                if ev.suggestion:
                    lines.append(f"Suggestion: {ev.suggestion}")
        
        return "\n".join(lines)
    
    def summarize(self) -> dict[str, Any]:
        """Get summary of the artifact for quick review.
        
        Returns:
            Summary dictionary.
        """
        total_steps = len(self.artifact.step_artifacts)
        successful = sum(
            1 for s in self.artifact.step_artifacts
            if json.loads(s.outcome_json).get("success", False)
        )
        
        return {
            "plan_id": self.artifact.plan_id,
            "final_status": self.artifact.final_status,
            "total_steps": total_steps,
            "successful_steps": successful,
            "failed_steps": total_steps - successful,
            "start_time": self.artifact.start_time,
            "end_time": self.artifact.end_time,
            "repo_fingerprint": self.artifact.repo_fingerprint,
        }
