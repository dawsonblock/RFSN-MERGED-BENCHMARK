"""Repair module - bug taxonomy and failure classification."""
from .taxonomy import RepairHypothesis, TAXONOMY, REPAIR_STRATEGIES
from .classifier import classify_failure, extract_error_signature

__all__ = [
    "RepairHypothesis",
    "TAXONOMY",
    "REPAIR_STRATEGIES",
    "classify_failure",
    "extract_error_signature",
]
