"""Artifact Log - Full traceability for plan execution.

Records plan JSON, each step's task spec, controller outcome payload,
and diff summary hash for complete auditability and replay.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schema import ControllerOutcome, ControllerTaskSpec, Plan

logger = logging.getLogger(__name__)


@dataclass
class StepArtifact:
    """Artifact for a single step execution."""
    
    step_id: str
    task_spec_json: str
    outcome_json: str
    diff_summary_hash: str
    wall_clock_ms: int
    timestamp: str
    files_touched: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_spec_json": self.task_spec_json,
            "outcome_json": self.outcome_json,
            "diff_summary_hash": self.diff_summary_hash,
            "wall_clock_ms": self.wall_clock_ms,
            "timestamp": self.timestamp,
            "files_touched": self.files_touched,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepArtifact:
        return cls(
            step_id=data["step_id"],
            task_spec_json=data["task_spec_json"],
            outcome_json=data["outcome_json"],
            diff_summary_hash=data["diff_summary_hash"],
            wall_clock_ms=data["wall_clock_ms"],
            timestamp=data["timestamp"],
            files_touched=data.get("files_touched", []),
        )


@dataclass
class PlanArtifact:
    """Full artifact for a plan run."""
    
    plan_id: str
    plan_json: str
    repo_fingerprint: str
    step_artifacts: list[StepArtifact]
    start_time: str
    end_time: str
    final_status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_json": self.plan_json,
            "repo_fingerprint": self.repo_fingerprint,
            "step_artifacts": [s.to_dict() for s in self.step_artifacts],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "final_status": self.final_status,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanArtifact:
        return cls(
            plan_id=data["plan_id"],
            plan_json=data["plan_json"],
            repo_fingerprint=data["repo_fingerprint"],
            step_artifacts=[StepArtifact.from_dict(s) for s in data.get("step_artifacts", [])],
            start_time=data["start_time"],
            end_time=data["end_time"],
            final_status=data["final_status"],
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> PlanArtifact:
        return cls.from_dict(json.loads(json_str))


class PlanArtifactLog:
    """Stores and retrieves plan artifacts for auditing and replay."""
    
    def __init__(self, output_dir: Path):
        """Initialize the artifact log.
        
        Args:
            output_dir: Base directory for storing artifacts.
        """
        self.output_dir = Path(output_dir) / "plan_artifacts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._active_artifacts: dict[str, dict[str, Any]] = {}
    
    def record_plan_start(
        self,
        plan: Plan,
        fingerprint: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start recording a plan execution.
        
        Args:
            plan: The plan being executed.
            fingerprint: Repo fingerprint hash.
            metadata: Optional additional metadata.
            
        Returns:
            Artifact ID for this recording.
        """
        artifact_id = f"{plan.plan_id}_{self._timestamp()}"
        logger.info("Recording plan start: %s", artifact_id)
        
        self._active_artifacts[artifact_id] = {
            "plan_id": plan.plan_id,
            "plan_json": plan.to_json(),
            "repo_fingerprint": fingerprint,
            # "step_artifacts": [],  <-- Removed, we don't hold in memory
            "start_time": self._now_iso(),
            "end_time": "",
            "final_status": "running",
            "metadata": metadata or {},
        }
        
        # Initialize storage files
        base_path = self.output_dir / artifact_id
        base_path.mkdir(exist_ok=True)
        
        # Write initial metadata
        self._write_meta(artifact_id, self._active_artifacts[artifact_id])
        
        return artifact_id
    
    def _write_meta(self, artifact_id: str, data: dict[str, Any]):
        """Write metadata to disk."""
        base_path = self.output_dir / artifact_id
        (base_path / "meta.json").write_text(json.dumps(data, indent=2))
    
    def record_step(
        self,
        artifact_id: str,
        spec: ControllerTaskSpec,
        outcome: ControllerOutcome,
        diff: str,
        elapsed_ms: int,
        files_touched: list[str] | None = None,
    ) -> None:
        """Record a step execution.
        
        Args:
            artifact_id: The artifact ID from record_plan_start.
            spec: The task spec sent to controller.
            outcome: The outcome from controller.
            diff: The diff produced (for hashing).
            elapsed_ms: Wall clock time for step.
            files_touched: Files modified in step.
        """
        if artifact_id not in self._active_artifacts:
            return
        
        step_artifact = StepArtifact(
            step_id=spec.step_id,
            task_spec_json=json.dumps(spec.to_dict()),
            outcome_json=json.dumps(outcome.to_dict()),
            diff_summary_hash=self._hash_diff(diff),
            wall_clock_ms=elapsed_ms,
            timestamp=self._now_iso(),
            files_touched=files_touched or [],
        )
        
        # Stream to JSONL
        base_path = self.output_dir / artifact_id
        with open(base_path / "steps.jsonl", "a") as f:
            f.write(json.dumps(step_artifact.to_dict()) + "\n")
    
    def record_qa_rejection(self, artifact_id: str, step_id: str, qa_result: Any) -> None:
        """Record QA rejection details."""
        if artifact_id not in self._active_artifacts:
            return
            
        # Append to metadata
        artifact_data = self._active_artifacts[artifact_id]
        if "qa_rejections" not in artifact_data["metadata"]:
            artifact_data["metadata"]["qa_rejections"] = []
            
        artifact_data["metadata"]["qa_rejections"].append({
            "step_id": step_id,
            "reasons": qa_result.rejection_reasons,
            "timestamp": self._now_iso(),
        })
    
    def finalize(self, artifact_id: str, status: str) -> Path | None:
        """Finalize and save the artifact.
        
        Args:
            artifact_id: The artifact ID.
            status: Final status (success, failed, halted, etc.)
            
        Returns:
            Path to saved artifact file, or None if not found.
        """
        if artifact_id not in self._active_artifacts:
            logger.warning("Attempted to finalize unknown artifact: %s", artifact_id)
            return None
        
        logger.info("Finalizing artifact: %s with status: %s", artifact_id, status)
        
        data = self._active_artifacts.pop(artifact_id)
        data["end_time"] = self._now_iso()
        data["final_status"] = status
        
        # Update metadata
        self._write_meta(artifact_id, data)
        
        return self.output_dir / artifact_id / "meta.json"
    
    def load(self, artifact_id: str) -> PlanArtifact | None:
        """Load an artifact by ID.
        
        Args:
            artifact_id: The artifact ID to load.
            
        Returns:
            PlanArtifact or None if not found.
        """
        base_path = self.output_dir / artifact_id
        meta_path = base_path / "meta.json"
        steps_path = base_path / "steps.jsonl"
        
        if not meta_path.exists():
            # Fallback for old single-file format
            old_path = self.output_dir / f"{artifact_id}.json"
            if old_path.exists():
                return PlanArtifact.from_json(old_path.read_text())
            return None
            
        data = json.loads(meta_path.read_text())
        
        # Load steps
        steps = []
        if steps_path.exists():
            with open(steps_path) as f:
                for line in f:
                    if line.strip():
                        steps.append(StepArtifact.from_dict(json.loads(line)))
                        
        return PlanArtifact(
            plan_id=data["plan_id"],
            plan_json=data["plan_json"],
            repo_fingerprint=data["repo_fingerprint"],
            step_artifacts=steps,
            start_time=data["start_time"],
            end_time=data["end_time"],
            final_status=data["final_status"],
            metadata=data["metadata"],
        )
    
    def list_artifacts(self, plan_id: str | None = None) -> list[str]:
        """List available artifact IDs.
        
        Args:
            plan_id: Optional filter by plan ID.
            
        Returns:
            List of artifact IDs.
        """
        artifacts = []
        artifacts = []
        # Check both directories and legacy files
        entries = list(self.output_dir.iterdir())
        
        for p in entries:
            aid = p.name
            if p.is_dir() and (p / "meta.json").exists():
                # New format
                if plan_id is None or aid.startswith(plan_id):
                    artifacts.append(aid)
            elif p.suffix == ".json" and not p.stem.startswith("meta"):
                 # Legacy format
                 if plan_id is None or p.stem.startswith(plan_id):
                     artifacts.append(p.stem)
                     
        return sorted(list(set(artifacts)), reverse=True)
    
    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()
    
    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def _hash_diff(diff: str) -> str:
        """Create a hash of the diff for comparison."""
        return hashlib.sha256(diff.encode()).hexdigest()[:16]
