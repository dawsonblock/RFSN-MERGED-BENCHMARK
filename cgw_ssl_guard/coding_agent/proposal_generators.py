"""Proposal generators for the coding agent.

Each generator implements a strategy for proposing actions to the
thalamic gate. Generators do NOT decide which action to take - they
only propose candidates that compete via the gate's scoring mechanism.

The key constraint is that generators are pure proposal sources:
- They analyze the current state
- They propose one or more candidates with saliency/urgency/surprise
- They never execute actions or modify state
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..types import Candidate
from .action_types import ActionPayload, CodingAction, ExecutionResult

if TYPE_CHECKING:
    from ...rfsn_controller.action_outcome_memory import ActionOutcomeMemory
    from ...rfsn_controller.planner_v2 import ControllerAdapter


@dataclass
class ProposalContext:
    """Context provided to generators for making proposals.
    
    This contains the current state of the coding workflow, including
    test results, patch history, and any other relevant information.
    """
    
    # Current state
    cycle_id: int
    last_action: Optional[CodingAction] = None
    last_result: Optional[ExecutionResult] = None
    
    # Test state
    tests_passing: bool = False
    failing_tests: List[str] = None
    test_output: str = ""
    
    # Patch state
    patches_applied: int = 0
    patches_reverted: int = 0
    current_diff: str = ""
    
    # Goals
    goal: str = ""
    subgoals: List[str] = None
    
    # Safety
    safety_triggered: bool = False
    safety_reason: str = ""
    
    def __post_init__(self):
        if self.failing_tests is None:
            self.failing_tests = []
        if self.subgoals is None:
            self.subgoals = []


class ProposalGenerator(ABC):
    """Abstract base class for proposal generators.
    
    Each generator analyzes the context and produces zero or more
    Candidate objects for the thalamic gate to arbitrate.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.proposals_made = 0
        self.proposals_selected = 0
    
    @abstractmethod
    def generate(self, context: ProposalContext) -> List[Candidate]:
        """Generate candidate proposals for the current context.
        
        Returns:
            List of Candidate objects with scored saliency/urgency/surprise.
        """
        pass
    
    def _make_candidate(
        self,
        action: CodingAction,
        saliency: float,
        urgency: float = 0.0,
        surprise: float = 0.0,
        parameters: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> Candidate:
        """Helper to create a Candidate with proper payload."""
        payload = ActionPayload(
            action=action,
            parameters=parameters or {},
            context=context or {},
            metadata={"generator": self.name, "timestamp": time.time()},
        )
        slot_id = f"{self.name}_{action.value}_{int(time.time() * 1e6)}"
        self.proposals_made += 1
        return Candidate(
            slot_id=slot_id,
            source_module=self.name,
            content_payload=payload.to_bytes(),
            saliency=saliency,
            urgency=urgency,
            surprise=surprise,
        )


class SafetyProposalGenerator(ProposalGenerator):
    """Generator that proposes ABORT when safety conditions are triggered.
    
    This generator always proposes with maximum urgency when safety
    is triggered, ensuring the gate will select ABORT.
    """
    
    def __init__(self):
        super().__init__("safety")
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        if context.safety_triggered:
            return [self._make_candidate(
                CodingAction.ABORT,
                saliency=1.0,
                urgency=1.0,  # Maximum urgency
                surprise=0.0,
                context={"reason": context.safety_reason},
            )]
        return []


class PlannerProposalGenerator(ProposalGenerator):
    """Generator that wraps PlannerV2 as a proposal source.
    
    Translates planner step recommendations into Candidates.
    """
    
    def __init__(self, adapter: Optional["ControllerAdapter"] = None):
        super().__init__("planner")
        self.adapter = adapter
        self._action_mapping = {
            "run_tests": CodingAction.RUN_TESTS,
            "analyze": CodingAction.ANALYZE_FAILURE,
            "generate_patch": CodingAction.GENERATE_PATCH,
            "apply_patch": CodingAction.APPLY_PATCH,
            "validate": CodingAction.VALIDATE,
            "finalize": CodingAction.FINALIZE,
        }
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        if self.adapter is None:
            return self._generate_heuristic(context)
        return self._generate_from_planner(context)
    
    def _generate_heuristic(self, context: ProposalContext) -> List[Candidate]:
        """Generate proposals using simple heuristics when no planner available."""
        candidates = []
        
        # If no tests run yet, propose running tests
        if context.last_action is None:
            candidates.append(self._make_candidate(
                CodingAction.RUN_TESTS,
                saliency=0.9,
                urgency=0.5,
                context={"reason": "initial_test_run"},
            ))
        
        # If tests failed, propose analysis
        elif context.last_action == CodingAction.RUN_TESTS and not context.tests_passing:
            candidates.append(self._make_candidate(
                CodingAction.ANALYZE_FAILURE,
                saliency=0.85,
                urgency=0.6,
                context={"failing_tests": context.failing_tests},
            ))
        
        # If analysis done, propose patch generation
        elif context.last_action == CodingAction.ANALYZE_FAILURE:
            candidates.append(self._make_candidate(
                CodingAction.GENERATE_PATCH,
                saliency=0.8,
                urgency=0.5,
            ))
        
        # If patch generated, propose applying it
        elif context.last_action == CodingAction.GENERATE_PATCH:
            candidates.append(self._make_candidate(
                CodingAction.APPLY_PATCH,
                saliency=0.85,
                urgency=0.6,
            ))
        
        # If patch applied, propose running tests again
        elif context.last_action == CodingAction.APPLY_PATCH:
            candidates.append(self._make_candidate(
                CodingAction.RUN_TESTS,
                saliency=0.9,
                urgency=0.7,
                context={"reason": "verify_patch"},
            ))
        
        # If tests now pass, propose finalization
        elif context.last_action == CodingAction.RUN_TESTS and context.tests_passing:
            candidates.append(self._make_candidate(
                CodingAction.FINALIZE,
                saliency=1.0,
                urgency=0.8,
                context={"reason": "tests_passing"},
            ))
        
        return candidates
    
    def _generate_from_planner(self, context: ProposalContext) -> List[Candidate]:
        """Generate proposals from PlannerV2 adapter."""
        # This would integrate with the actual PlannerV2 ControllerAdapter
        # For now, fall back to heuristics
        return self._generate_heuristic(context)


class MemoryProposalGenerator(ProposalGenerator):
    """Generator that uses action outcome memory to bias proposals.
    
    Queries historical outcomes for similar situations and adjusts
    saliency based on past success rates.
    """
    
    def __init__(self, memory: Optional["ActionOutcomeMemory"] = None):
        super().__init__("memory")
        self.memory = memory
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        if self.memory is None:
            return []
        
        candidates = []
        
        # Query memory for similar failure patterns
        if context.failing_tests and context.last_action == CodingAction.ANALYZE_FAILURE:
            # Check if we've seen similar failures before
            # This would query the ActionOutcomeMemory for patterns
            # and suggest actions that worked previously
            pass
        
        return candidates


class AnalyzerProposalGenerator(ProposalGenerator):
    """Generator that analyzes test output to make targeted proposals.
    
    Uses pattern matching on test output to determine the best
    next action with high confidence.
    """
    
    def __init__(self):
        super().__init__("analyzer")
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        candidates = []
        
        if context.test_output and not context.tests_passing:
            # Analyze the test output for patterns
            output_lower = context.test_output.lower()
            
            # High surprise for unexpected errors
            if "import error" in output_lower or "modulenotfounderror" in output_lower:
                candidates.append(self._make_candidate(
                    CodingAction.INSPECT_FILES,
                    saliency=0.7,
                    urgency=0.4,
                    surprise=0.8,  # Import errors are often surprising
                    context={"pattern": "import_error"},
                ))
            
            # Syntax errors are urgent
            if "syntaxerror" in output_lower:
                candidates.append(self._make_candidate(
                    CodingAction.ANALYZE_TRACEBACK,
                    saliency=0.8,
                    urgency=0.9,  # Syntax errors block everything
                    context={"pattern": "syntax_error"},
                ))
        
        return candidates


class IdleProposalGenerator(ProposalGenerator):
    """Generator that proposes IDLE when no other action is appropriate.
    
    Acts as a fallback with very low saliency so other generators
    can always win if they have proposals.
    """
    
    def __init__(self):
        super().__init__("idle")
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        return [self._make_candidate(
            CodingAction.IDLE,
            saliency=0.01,  # Very low - any real action should win
            urgency=0.0,
            surprise=0.0,
        )]
