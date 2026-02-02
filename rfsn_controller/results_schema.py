"""Canonical results artifact schema for RFSN Controller.

This module defines the standard format for all run artifacts,
ensuring consistency across evaluation and production runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

@dataclass
class RunSummary:
    """Summary of a single task execution."""
    task_id: str
    instance_id: str
    repo: str
    base_commit: str
    status: str  # PASS, FAIL, ERROR, TIMEOUT, REJECTED
    passed: bool
    runtime_s: float
    attempts: int
    gate_rejections: int
    security_violations: int
    final_patch: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunArtifact:
    """Complete artifact for a run, including events and environment."""
    summary: RunSummary
    events: List[Dict[str, Any]] = field(default_factory=list)
    env_capture: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)

    def save(self, output_dir: Path):
        """Save artifact to disk in canonical format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save summary.json
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(self.summary), f, indent=2)
            
        # 2. Save events.jsonl
        events_path = output_dir / "events.jsonl"
        with open(events_path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
                
        # 3. Save full artifact.json
        artifact_path = output_dir / "artifact.json"
        with open(artifact_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
            
        return artifact_path

def capture_env() -> Dict[str, Any]:
    """Capture relevant environment metadata."""
    import os
    import sys
    import platform
    
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version,
        "rfsn_version": "1.4.3", # From pyproject.toml
        "env_vars": {
            k: "SET" for k in os.environ 
            if "API_KEY" in k or "SECRET" in k or "TOKEN" in k
        }
    }
