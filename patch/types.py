"""Patch data types and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class PatchStatus(Enum):
    """Status of a patch."""
    
    GENERATED = "generated"
    SCORED = "scored"
    APPLIED = "applied"
    TESTED = "tested"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PatchStrategy(Enum):
    """Strategy used to generate patch."""
    
    DIRECT_FIX = "direct_fix"          # Direct fix based on localization
    TEST_DRIVEN = "test_driven"        # Generate to pass failing tests
    HYPOTHESIS = "hypothesis"          # Try a hypothesis about the bug
    INCREMENTAL = "incremental"        # Build on previous patch
    ENSEMBLE = "ensemble"              # Combine multiple approaches


@dataclass
class FileDiff:
    """A diff for a single file."""
    
    file_path: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    unified_diff: str = ""
    
    # Line-level changes
    added_lines: List[int] = field(default_factory=list)
    removed_lines: List[int] = field(default_factory=list)
    modified_lines: List[int] = field(default_factory=list)


@dataclass
class Patch:
    """A code patch with metadata."""
    
    # Identity
    patch_id: str
    strategy: PatchStrategy
    
    # Content
    diff_text: str
    file_diffs: List[FileDiff] = field(default_factory=list)
    
    # Metadata
    status: PatchStatus = PatchStatus.GENERATED
    score: float = 0.0
    confidence: float = 0.0
    
    # Generation context
    localization_hits: List[str] = field(default_factory=list)
    rationale: str = ""
    generation_method: str = ""
    
    # Evaluation results
    syntax_valid: bool = False
    imports_valid: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    test_output: str = ""
    
    # Scoring components
    static_score: float = 0.0
    test_score: float = 0.0
    diff_risk_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patch_id": self.patch_id,
            "strategy": self.strategy.value,
            "diff_text": self.diff_text,
            "status": self.status.value,
            "score": self.score,
            "confidence": self.confidence,
            "localization_hits": self.localization_hits,
            "rationale": self.rationale,
            "generation_method": self.generation_method,
            "syntax_valid": self.syntax_valid,
            "imports_valid": self.imports_valid,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "static_score": self.static_score,
            "test_score": self.test_score,
            "diff_risk_score": self.diff_risk_score,
            "metadata": self.metadata,
        }


@dataclass
class PatchCandidate:
    """A candidate patch with generation context."""
    
    patch: Patch
    priority: float = 0.0
    generation_time: float = 0.0
    tokens_used: int = 0
    
    # Parent/child relationships for incremental patching
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


@dataclass
class PatchGenerationRequest:
    """Request for patch generation."""
    
    # Context
    problem_statement: str
    repo_dir: str
    localization_hits: List[Dict[str, Any]]
    
    # Optional context
    failing_tests: List[str] = field(default_factory=list)
    traceback: Optional[str] = None
    previous_patches: List[Patch] = field(default_factory=list)
    
    # Generation parameters
    strategy: PatchStrategy = PatchStrategy.DIRECT_FIX
    max_patches: int = 5
    max_files_per_patch: int = 3
    max_lines_per_file: int = 50
    
    # Constraints
    allowed_files: Optional[List[str]] = None
    forbidden_patterns: List[str] = field(default_factory=list)


@dataclass
class PatchGenerationResult:
    """Result of patch generation."""
    
    candidates: List[PatchCandidate]
    total_generated: int = 0
    generation_time: float = 0.0
    tokens_used: int = 0
    errors: List[str] = field(default_factory=list)
