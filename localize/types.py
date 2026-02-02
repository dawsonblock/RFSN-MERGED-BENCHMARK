"""Localization types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass
class LocalizationHit:
    """A localized file/span with evidence."""
    
    # Core fields (support both old and new naming)
    file_path: str
    line_start: int
    line_end: int
    score: float  # 0.0 to 1.0
    evidence: str
    method: str  # trace, ripgrep, embedding, symbol
    
    # Optional metadata
    snippet: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy compatibility properties
    @property
    def file(self) -> str:
        """Legacy file field."""
        return self.file_path
    
    @property
    def span(self) -> Tuple[int, int]:
        """Legacy span field."""
        return (self.line_start, self.line_end)
    
    @property
    def evidence_type(self) -> str:
        """Legacy evidence_type field."""
        return self.method
    
    @property
    def evidence_text(self) -> str:
        """Legacy evidence_text field."""
        return self.evidence
    
    @property
    def why(self) -> str:
        """Legacy why field."""
        return self.evidence
