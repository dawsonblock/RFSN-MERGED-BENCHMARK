"""Regression Memory Firewall - Block toxic patterns.

Prevents the system from re-applying patches that previously caused
regressions or critical failures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToxicSignature:
    """A signature of a toxic patch."""
    signature_hash: str
    failure_type: str
    revert_count: int = 1
    files: list[str] = field(default_factory=list)
    timestamp: float = 0.0


class RegressionFirewall:
    """Blocks patches that match known toxic signatures."""
    
    def __init__(self, storage_path: Path | None = None):
        self._storage_path = storage_path
        self._signatures: dict[str, ToxicSignature] = {}
        self._file_index: dict[str, set[str]] = {} # Reverse index: filename -> {sig_hashes}
        
        if storage_path and storage_path.exists():
            self._load()
            
    def compute_signature(self, files: list[str], diff_content: str) -> str:
        """Compute a hash signature for a patch."""
        # Clean diff content (ignore line numbers/timestamps)
        # This is a simplified signature
        content = "".join(sorted(files)) + diff_content
        return hashlib.sha256(content.encode()).hexdigest()
        
    def is_toxic(self, files: list[str], diff_content: str) -> bool:
        """Check if a patch signature is known to be toxic."""
        sig = self.compute_signature(files, diff_content)
        return sig in self._signatures
        
    def record_toxicity(
        self,
        files: list[str],
        diff_content: str,
        failure_type: str,
    ):
        """Record a patch as toxic.
        
        Args:
            files: List of modified files.
            diff_content: The patch content.
            failure_type: The type of failure it caused.
        """
        import time
        sig = self.compute_signature(files, diff_content)
        
        if sig in self._signatures:
            self._signatures[sig].revert_count += 1
            self._signatures[sig].timestamp = time.time()
        else:
            self._signatures[sig] = ToxicSignature(
                signature_hash=sig,
                failure_type=failure_type,
                revert_count=1,
                files=files,
                timestamp=time.time(),
            )
            # Update index
            for f in files:
                if f not in self._file_index:
                    self._file_index[f] = set()
                self._file_index[f].add(sig)
            
        self._save()
        
    def _save(self):
        """Persist firewall rules."""
        if not self._storage_path:
            return
            
        data = {
            sig: {
                "signature_hash": t.signature_hash,
                "failure_type": t.failure_type,
                "revert_count": t.revert_count,
                "files": t.files,
                "timestamp": t.timestamp
            }
            for sig, t in self._signatures.items()
        }
        
        with open(self._storage_path, "w") as f:
            json.dump(data, f)
            
    def _load(self):
        """Load firewall rules."""
        if not self._storage_path or not self._storage_path.exists():
            return
            
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
                
            for sig, t_dict in data.items():
                self._signatures[sig] = ToxicSignature(
                    signature_hash=sig,
                    failure_type=t_dict["failure_type"],
                    revert_count=t_dict.get("revert_count", 1),
                    timestamp=t_dict.get("timestamp", 0.0)
                )
                # Rebuild index
                for f in t_dict.get("files", []):
                    if f not in self._file_index:
                        self._file_index[f] = set()
                    self._file_index[f].add(sig)
        except Exception:
            pass

    def get_toxic_history(self, allowed_files: list[str]) -> list[str]:
        """Get summary of toxic history for files.
        
        Args:
            allowed_files: Files step is allowed to touch.
            
        Returns:
            List of warning strings.
        """
        warnings = []
        warnings = []
        # Optimized lookup using reverse index
        checked_signatures = set()
        
        for f in allowed_files:
            if f in self._file_index:
                for sig_hash in self._file_index[f]:
                    if sig_hash in checked_signatures:
                        continue
                    checked_signatures.add(sig_hash)
                    
                    sig = self._signatures.get(sig_hash)
                    if sig:
                        warnings.append(
                            f"Prior regression in {sig.files} (type: {sig.failure_type})"
                        )
                        
        return list(set(warnings))
