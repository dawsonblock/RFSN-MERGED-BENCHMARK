"""
from __future__ import annotations
RFSN Planner v5 - SWE-bench Optimized Coding Planner

A proposal-only coding planner that operates under strict RFSN authority model.
Never executes code, never bypasses safety, only generates structured proposals.

Key Principles:
- Untrusted by design
- Serial execution only
- Evidence-driven proposals
- Gate-first architecture
- Minimal verifiable changes

Architecture:
- ProposalPlanner: Generates individual proposals
- MetaPlanner: Strategy layer above proposal generation
- ScoringEngine: Evaluates proposal quality
- StateTracker: Maintains planning state
"""

from .meta_planner import MetaPlanner, PlannerState
from .planner import ProposalPlanner
from .proposal import ActionType, ExpectedEffect, Proposal, ProposalIntent, RiskLevel, Target, TestExpectation
from .scoring import CandidateScore, ScoringEngine
from .state_tracker import GateRejectionType, HypothesisOutcome, StateTracker

__version__ = "5.0.0"

__all__ = [
    # Core types
    "Proposal",
    "ProposalIntent",
    "ActionType",
    "ExpectedEffect",
    "RiskLevel",
    "Target",
    "TestExpectation",
    # Planners
    "ProposalPlanner",
    "MetaPlanner",
    "PlannerState",
    # Scoring
    "ScoringEngine",
    "CandidateScore",
    # State tracking
    "StateTracker",
    "HypothesisOutcome",
    "GateRejectionType",
]
