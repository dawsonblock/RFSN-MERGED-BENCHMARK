"""
Test Artifacts Capture System

Captures and stores test execution artifacts:
- Test output (stdout/stderr)
- Stack traces and error messages
- Test timing and performance data
- Coverage reports
- Log files
- Screenshots (for UI tests)
"""

from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestArtifact:
    """Single test artifact"""
    artifact_type: str  # stdout, stderr, traceback, coverage, log
    content: str
    size_bytes: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactCollection:
    """Collection of artifacts from a test run"""
    run_id: str
    timestamp: str
    stage: str
    artifacts: List[TestArtifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArtifactCapture:
    """Capture and store test artifacts"""
    
    def __init__(self, storage_dir: str = ".rfsn/artifacts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_from_stage(
        self,
        run_id: str,
        stage: str,
        stdout: str,
        stderr: str,
        additional_artifacts: Optional[Dict[str, str]] = None
    ) -> ArtifactCollection:
        """Capture artifacts from a test stage"""
        
        timestamp = datetime.utcnow().isoformat()
        
        artifacts = []
        
        # Capture stdout
        if stdout:
            artifacts.append(TestArtifact(
                artifact_type="stdout",
                content=stdout,
                size_bytes=len(stdout.encode('utf-8')),
                timestamp=timestamp
            ))
        
        # Capture stderr
        if stderr:
            artifacts.append(TestArtifact(
                artifact_type="stderr",
                content=stderr,
                size_bytes=len(stderr.encode('utf-8')),
                timestamp=timestamp
            ))
        
        # Capture additional artifacts
        if additional_artifacts:
            for artifact_type, content in additional_artifacts.items():
                artifacts.append(TestArtifact(
                    artifact_type=artifact_type,
                    content=content,
                    size_bytes=len(content.encode('utf-8')),
                    timestamp=timestamp
                ))
        
        collection = ArtifactCollection(
            run_id=run_id,
            timestamp=timestamp,
            stage=stage,
            artifacts=artifacts,
            metadata={
                "total_size": sum(a.size_bytes for a in artifacts),
                "artifact_count": len(artifacts)
            }
        )
        
        # Store artifacts
        self._store_collection(collection)
        
        logger.info(
            f"Captured {len(artifacts)} artifacts for run {run_id} "
            f"stage {stage} ({collection.metadata['total_size']} bytes)"
        )
        
        return collection
    
    def _store_collection(self, collection: ArtifactCollection):
        """Store artifact collection to disk"""
        
        # Create directory for this run
        run_dir = self.storage_dir / collection.run_id
        run_dir.mkdir(exist_ok=True)
        
        # Store each artifact
        for i, artifact in enumerate(collection.artifacts):
            artifact_file = run_dir / f"{collection.stage}_{artifact.artifact_type}_{i}.txt"
            
            # Compress large artifacts
            if artifact.size_bytes > 100_000:  # 100KB threshold
                artifact_file = artifact_file.with_suffix('.txt.gz')
                with gzip.open(artifact_file, 'wt') as f:
                    f.write(artifact.content)
            else:
                with open(artifact_file, 'w') as f:
                    f.write(artifact.content)
        
        # Store collection metadata
        meta_file = run_dir / f"{collection.stage}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(asdict(collection), f, indent=2, default=str)
    
    def load_collection(
        self,
        run_id: str,
        stage: str
    ) -> Optional[ArtifactCollection]:
        """Load artifact collection from storage"""
        
        run_dir = self.storage_dir / run_id
        meta_file = run_dir / f"{stage}_metadata.json"
        
        if not meta_file.exists():
            logger.warning(f"No artifacts found for run {run_id} stage {stage}")
            return None
        
        try:
            with open(meta_file) as f:
                data = json.load(f)
            
            # Reconstruct collection
            collection = ArtifactCollection(**data)
            
            # Load artifact content
            for artifact in collection.artifacts:
                # Artifacts stored separately would be loaded here
                pass
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            return None
    
    def cleanup_old_artifacts(self, keep_days: int = 7):
        """Clean up artifacts older than specified days"""
        
        cutoff = datetime.utcnow().timestamp() - (keep_days * 24 * 60 * 60)
        
        removed_count = 0
        removed_size = 0
        
        for run_dir in self.storage_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # Check modification time
            if run_dir.stat().st_mtime < cutoff:
                size = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
                shutil.rmtree(run_dir)
                removed_count += 1
                removed_size += size
        
        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} artifact collections "
                f"({removed_size / 1024 / 1024:.1f} MB)"
            )
    
    def get_artifact_stats(self) -> Dict[str, Any]:
        """Get statistics about stored artifacts"""
        
        total_runs = len(list(self.storage_dir.iterdir()))
        total_size = sum(
            f.stat().st_size
            for f in self.storage_dir.rglob('*')
            if f.is_file()
        )
        
        return {
            "total_runs": total_runs,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "storage_dir": str(self.storage_dir)
        }


def extract_stack_traces(content: str) -> List[str]:
    """Extract stack traces from test output"""
    
    traces = []
    in_trace = False
    current_trace = []
    
    for line in content.split('\n'):
        # Start of traceback
        if 'Traceback' in line or 'Stack trace' in line:
            in_trace = True
            current_trace = [line]
        elif in_trace:
            current_trace.append(line)
            # End of traceback (empty line or new test)
            if not line.strip() or line.startswith('====='):
                traces.append('\n'.join(current_trace))
                in_trace = False
                current_trace = []
    
    return traces


def extract_error_messages(content: str) -> List[str]:
    """Extract error messages from test output"""
    
    errors = []
    
    for line in content.split('\n'):
        # Common error patterns
        if any(keyword in line for keyword in ['Error:', 'Exception:', 'AssertionError:', 'FAILED']):
            errors.append(line.strip())
    
    return errors


if __name__ == "__main__":
    # Test artifact capture
    capture = ArtifactCapture()
    
    collection = capture.capture_from_stage(
        run_id="test_run_123",
        stage="baseline",
        stdout="Test output here...",
        stderr="Some warnings...",
        additional_artifacts={
            "coverage": "Coverage: 85%",
            "timing": "Total: 1.5s"
        }
    )
    
    print(f"Captured {len(collection.artifacts)} artifacts")
    
    # Stats
    stats = capture.get_artifact_stats()
    print(f"Storage stats: {stats}")
