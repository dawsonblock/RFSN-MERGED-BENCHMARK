"""Evidence pack export module for RFSN controller.
from __future__ import annotations

Creates structured evidence packs documenting fixes.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class EvidencePackConfig:
    """Configuration for evidence pack export."""
    output_dir: str
    include_diffs: bool = True
    include_logs: bool = True
    include_screenshots: bool = False
    max_log_lines: int = 1000
    compress: bool = False


@dataclass 
class EvidenceItem:
    """A single piece of evidence."""
    type: str  # "diff", "test_output", "log", "screenshot"
    path: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidencePackExporter:
    """Exports evidence packs for debugging and auditing."""
    
    def __init__(self, config: EvidencePackConfig):
        self.config = config
        self.items: list[EvidenceItem] = []
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def add_diff(self, file_path: str, diff_content: str, description: str = "") -> None:
        """Add a diff to the evidence pack."""
        if not self.config.include_diffs:
            return
        
        item = EvidenceItem(
            type="diff",
            path=file_path,
            description=description or f"Diff for {file_path}",
            metadata={"diff": diff_content}
        )
        self.items.append(item)
    
    def add_test_output(self, test_name: str, output: str, passed: bool) -> None:
        """Add test output to the evidence pack."""
        item = EvidenceItem(
            type="test_output",
            path=test_name,
            description=f"Test {'passed' if passed else 'failed'}: {test_name}",
            metadata={"output": output[:self.config.max_log_lines], "passed": passed}
        )
        self.items.append(item)
    
    def add_log(self, log_name: str, content: str) -> None:
        """Add a log file to the evidence pack."""
        if not self.config.include_logs:
            return
        
        item = EvidenceItem(
            type="log",
            path=log_name,
            description=f"Log: {log_name}",
            metadata={"content": content[:self.config.max_log_lines * 100]}
        )
        self.items.append(item)
    
    def export(
        self,
        sandbox_root: str = "",
        log_dir: str = "",
        baseline_output: str = "",
        final_output: str = "",
        winner_diff: str = "",
        state: dict[str, Any] | None = None,
        command_log: list[Any] | None = None,
        run_id: str = "",
    ) -> str:
        """Export the evidence pack to the output directory."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        pack_dir = os.path.join(self.config.output_dir, f"evidence_{timestamp}")
        os.makedirs(pack_dir, exist_ok=True)
        
        # Export manifest
        manifest = {
            "created_at": timestamp,
            "item_count": len(self.items),
            "items": []
        }
        
        for i, item in enumerate(self.items):
            item_data = {
                "index": i,
                "type": item.type,
                "path": item.path,
                "description": item.description,
                "timestamp": item.timestamp.isoformat(),
            }
            manifest["items"].append(item_data)
            
            # Write item content
            item_file = os.path.join(pack_dir, f"item_{i:03d}_{item.type}.json")
            with open(item_file, "w") as f:
                json.dump({**item_data, "metadata": item.metadata}, f, indent=2)
        
        # Write manifest
        manifest_path = os.path.join(pack_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return pack_dir
    
    def clear(self) -> None:
        """Clear all evidence items."""
        self.items = []
