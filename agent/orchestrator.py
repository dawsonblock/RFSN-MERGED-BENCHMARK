"""Episode Orchestrator - Wire LLM, Test Runner, and Failure Triage into Episode Loop.

This module integrates:
1. LLM patch generation (llm/patch_generator.py)
2. Staged test runner (runner/tests.py)
3. Failure triage (triage/failures.py)
4. Patch scoring/minimization (patch/score.py, patch/minimize.py)

Into the main episode loop, replacing the abstract functions with real implementations.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from .types import (
    AgentState,
    Proposal,
    GateDecision,
    ExecResult,
    Phase,
    Evidence,
)
from .profiles import Profile
from .loop import run_episode

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from localize import MultiLayerLocalizer, localize_issue
from localize.types import LocalizationHit
from patch.types import (
    PatchGenerationRequest,
    PatchStrategy,
    PatchCandidate,
    Patch,
)
from patch.gen import PatchGenerator
from patch.score import score_patches, PatchScore
from patch.minimize import minimize_patch, PatchMinimizer
from patch.apply import apply_patch_safe, PatchApplier
from runner.tests import run_staged_tests, TestStageConfig
from triage.failures import triage_failure, FailureType

try:
    from llm.client import get_llm_client, LLMClient
    from llm.patch_generator import LLMPatchGenerator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    get_llm_client = None
    LLMPatchGenerator = None

try:
    from rfsn_controller.structured_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class EpisodeOrchestrator:
    """Orchestrates complete episode execution with all components wired."""
    
    def __init__(
        self,
        profile: Profile,
        repo_path: str,
        test_command: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4-turbo-preview",
        use_docker: bool = False,
    ):
        """Initialize orchestrator.
        
        Args:
            profile: Agent profile with constraints
            repo_path: Path to repository
            test_command: Test command (e.g., "pytest tests/")
            llm_provider: LLM provider (openai, anthropic, deepseek)
            llm_model: LLM model name
            use_docker: Use Docker for test isolation
        """
        self.profile = profile
        self.repo_path = Path(repo_path)
        self.test_command = test_command or "pytest tests/"
        self.use_docker = use_docker
        
        # Initialize components
        self.localizer = MultiLayerLocalizer(repo_path=str(self.repo_path))
        
        # Initialize LLM client
        if LLM_AVAILABLE:
            try:
                self.llm_client = get_llm_client(
                    provider=llm_provider,
                    model=llm_model,
                )
                self.patch_generator = LLMPatchGenerator(llm_client=self.llm_client)
                logger.info(f"LLM client initialized: {llm_provider}/{llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}, using fallback")
                self.llm_client = None
                self.patch_generator = PatchGenerator()
        else:
            logger.warning("LLM not available, using mock patch generator")
            self.llm_client = None
            self.patch_generator = PatchGenerator()
        
        self.patch_minimizer = PatchMinimizer()
        self.patch_applier = PatchApplier(repo_path=str(self.repo_path))
        
        # State tracking
        self.current_patches: List[PatchCandidate] = []
        self.baseline_tests_run = False
        self.baseline_result = None
    
    def run(self, state: AgentState) -> AgentState:
        """Run complete episode.
        
        Args:
            state: Initial agent state
            
        Returns:
            Final agent state
        """
        logger.info(f"Starting orchestrated episode: {state.task_id}")
        
        # Wire episode loop with real functions
        final_state = run_episode(
            profile=self.profile,
            state=state,
            propose_fn=self._propose,
            gate_fn=self._gate,
            exec_fn=self._execute,
        )
        
        logger.info(f"Episode complete: {state.task_id}, phase={final_state.phase}")
        return final_state
    
    def _propose(self, profile: Profile, state: AgentState) -> Proposal:
        """Generate next proposal based on current phase.
        
        This is the PLANNER function that feeds the loop.
        
        Args:
            profile: Agent profile
            state: Current state
            
        Returns:
            Next proposal
        """
        logger.info(f"Proposing action for phase: {state.phase.value}")
        
        if state.phase == Phase.INGEST:
            return self._propose_ingest(state)
        
        elif state.phase == Phase.LOCALIZE:
            return self._propose_localize(state)
        
        elif state.phase == Phase.PLAN:
            return self._propose_plan(state)
        
        elif state.phase == Phase.PATCH_CANDIDATES:
            return self._propose_patch_candidates(state)
        
        elif state.phase == Phase.TEST_STAGE:
            return self._propose_test_stage(state)
        
        elif state.phase == Phase.DIAGNOSE:
            return self._propose_diagnose(state)
        
        elif state.phase == Phase.MINIMIZE:
            return self._propose_minimize(state)
        
        elif state.phase == Phase.FINALIZE:
            return self._propose_finalize(state)
        
        else:
            # Fallback
            return Proposal(
                kind="inspect",
                rationale=f"Inspect state at phase {state.phase.value}",
                inputs={"phase": state.phase.value},
            )
    
    def _propose_ingest(self, state: AgentState) -> Proposal:
        """Propose ingesting problem statement."""
        return Proposal(
            kind="inspect",
            rationale="Ingest problem statement and repository snapshot",
            inputs={"action": "ingest", "repo_path": str(self.repo_path)},
        )
    
    def _propose_localize(self, state: AgentState) -> Proposal:
        """Propose localization."""
        problem_statement = state.notes.get("problem_statement", "")
        
        return Proposal(
            kind="search",
            rationale="Localize relevant files using multi-layer approach",
            inputs={
                "action": "localize",
                "problem_statement": problem_statement,
                "repo_path": str(self.repo_path),
            },
        )
    
    def _propose_plan(self, state: AgentState) -> Proposal:
        """Propose planning based on localization."""
        return Proposal(
            kind="inspect",
            rationale="Plan patch generation based on localization hits",
            inputs={
                "action": "plan",
                "localization_hits": len(state.localization_hits),
            },
        )
    
    def _propose_patch_candidates(self, state: AgentState) -> Proposal:
        """Propose generating patch candidates."""
        return Proposal(
            kind="edit",
            rationale="Generate patch candidates using LLM",
            inputs={
                "action": "generate_patches",
                "localization_hits": state.localization_hits,
                "problem_statement": state.notes.get("problem_statement", ""),
                "strategy": state.notes.get("patch_strategy", "direct_fix"),
            },
        )
    
    def _propose_test_stage(self, state: AgentState) -> Proposal:
        """Propose running staged tests."""
        return Proposal(
            kind="run_tests",
            rationale="Run staged tests to validate patch",
            inputs={
                "action": "staged_tests",
                "test_command": self.test_command,
                "use_docker": self.use_docker,
            },
        )
    
    def _propose_diagnose(self, state: AgentState) -> Proposal:
        """Propose diagnosing test failures."""
        return Proposal(
            kind="inspect",
            rationale="Diagnose test failures and update hypotheses",
            inputs={
                "action": "diagnose",
                "failures": state.last_failures,
            },
        )
    
    def _propose_minimize(self, state: AgentState) -> Proposal:
        """Propose minimizing successful patch."""
        return Proposal(
            kind="edit",
            rationale="Minimize patch using delta debugging",
            inputs={
                "action": "minimize_patch",
            },
        )
    
    def _propose_finalize(self, state: AgentState) -> Proposal:
        """Propose finalizing the episode."""
        return Proposal(
            kind="finalize",
            rationale="Finalize patch and create PR/commit",
            inputs={
                "action": "finalize",
                "create_pr": state.notes.get("create_pr", False),
                "commit_changes": state.notes.get("commit_changes", True),
            },
        )
    
    def _gate(
        self, profile: Profile, state: AgentState, proposal: Proposal
    ) -> GateDecision:
        """Validate proposal through gate.
        
        For now, we use simple validation. Real gate would check:
        - Phase constraints
        - File constraints
        - Risk constraints
        
        Args:
            profile: Agent profile
            state: Current state
            proposal: Proposal to validate
            
        Returns:
            Gate decision
        """
        # Simple gate: accept most proposals, reject only obvious violations
        
        # Check budget constraints
        from .loop import check_budgets
        within_budget, reason = check_budgets(state, profile)
        if not within_budget:
            return GateDecision(
                accept=False,
                reason=f"Budget exceeded: {reason}",
                constraints={"budget": reason},
            )
        
        # Check phase-action compatibility
        valid_actions = {
            Phase.INGEST: ["inspect"],
            Phase.LOCALIZE: ["search"],
            Phase.PLAN: ["inspect"],
            Phase.PATCH_CANDIDATES: ["edit"],
            Phase.TEST_STAGE: ["run_tests"],
            Phase.DIAGNOSE: ["inspect"],
            Phase.MINIMIZE: ["edit"],
            Phase.FINALIZE: ["finalize"],
        }
        
        if proposal.kind not in valid_actions.get(state.phase, []):
            return GateDecision(
                accept=False,
                reason=f"Action {proposal.kind} not allowed in phase {state.phase.value}",
                constraints={"phase": state.phase.value, "action": proposal.kind},
            )
        
        # Accept
        return GateDecision(accept=True, reason="Proposal accepted")
    
    def _execute(
        self, profile: Profile, state: AgentState, proposal: Proposal
    ) -> ExecResult:
        """Execute approved proposal.
        
        This is the CONTROLLER function that performs actions.
        
        Args:
            profile: Agent profile
            state: Current state
            proposal: Approved proposal
            
        Returns:
            Execution result
        """
        logger.info(f"Executing {proposal.kind}: {proposal.rationale[:100]}")
        
        try:
            action = proposal.inputs.get("action", proposal.kind)
            
            if action == "ingest":
                return self._exec_ingest(state, proposal)
            
            elif action == "localize":
                return self._exec_localize(state, proposal)
            
            elif action == "plan":
                return self._exec_plan(state, proposal)
            
            elif action == "generate_patches":
                return self._exec_generate_patches(state, proposal)
            
            elif action == "staged_tests":
                return self._exec_staged_tests(state, proposal)
            
            elif action == "diagnose":
                return self._exec_diagnose(state, proposal)
            
            elif action == "minimize_patch":
                return self._exec_minimize_patch(state, proposal)
            
            elif action == "finalize":
                return self._exec_finalize(state, proposal)
            
            else:
                return ExecResult(
                    status="fail",
                    summary=f"Unknown action: {action}",
                )
        
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecResult(
                status="fail",
                summary=f"Execution error: {str(e)}",
            )
    
    def _exec_ingest(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute ingest phase."""
        logger.info("Ingesting problem statement")
        
        # Store problem statement in state
        problem_statement = state.notes.get("problem_statement", "")
        
        return ExecResult(
            status="ok",
            summary="Problem statement ingested",
            metrics={"problem_length": len(problem_statement)},
        )
    
    def _exec_localize(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute localization."""
        logger.info("Running multi-layer localization")
        
        problem_statement = proposal.inputs.get("problem_statement", "")
        
        # Run localization
        hits = localize_issue(
            problem_statement=problem_statement,
            repo_path=str(self.repo_path),
            limit=20,
        )
        
        # Store in state
        state.localization_hits = [
            {
                "file_path": hit.file_path,
                "line_start": hit.line_start,
                "line_end": hit.line_end,
                "score": hit.score,
                "evidence": hit.evidence,
                "method": hit.method,
            }
            for hit in hits
        ]
        
        logger.info(f"Localization found {len(hits)} hits")
        
        return ExecResult(
            status="ok",
            summary=f"Localized {len(hits)} candidate regions",
            metrics={"localization_hits": len(hits)},
        )
    
    def _exec_plan(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute planning."""
        logger.info("Planning patch generation")
        
        # For now, just mark as complete
        # Real planner would decompose into sub-tasks
        
        return ExecResult(
            status="ok",
            summary="Planning complete",
            metrics={"plan_ready": True},
        )
    
    def _exec_generate_patches(
        self, state: AgentState, proposal: Proposal
    ) -> ExecResult:
        """Execute patch generation."""
        logger.info("Generating patches with LLM")
        
        # Convert localization hits to proper format
        hits = []
        for hit_dict in proposal.inputs.get("localization_hits", []):
            hits.append(
                LocalizationHit(
                    file_path=hit_dict["file_path"],
                    line_start=hit_dict["line_start"],
                    line_end=hit_dict["line_end"],
                    score=hit_dict["score"],
                    evidence=hit_dict["evidence"],
                    method=hit_dict["method"],
                )
            )
        
        # Create generation request
        strategy_map = {
            "direct_fix": PatchStrategy.DIRECT_FIX,
            "test_driven": PatchStrategy.TEST_DRIVEN,
            "hypothesis": PatchStrategy.HYPOTHESIS,
            "incremental": PatchStrategy.INCREMENTAL,
            "ensemble": PatchStrategy.ENSEMBLE,
        }
        
        strategy = strategy_map.get(
            proposal.inputs.get("strategy", "direct_fix"),
            PatchStrategy.DIRECT_FIX,
        )
        
        request = PatchGenerationRequest(
            problem_statement=proposal.inputs.get("problem_statement", ""),
            localization_hits=hits,
            repo_dir=str(self.repo_path),
            strategy=strategy,
            max_patches=5,
        )
        
        # Generate patches
        result = self.patch_generator.generate(request)
        
        # Store candidates
        self.current_patches = result.candidates
        
        # Score patches
        if self.current_patches:
            scores = score_patches(
                patches=self.current_patches,
                repo_path=str(self.repo_path),
            )
            
            # Sort by score
            sorted_candidates = sorted(
                zip(self.current_patches, scores),
                key=lambda x: x[1].total_score,
                reverse=True,
            )
            self.current_patches = [c for c, _ in sorted_candidates]
            
            # Apply top patch
            if self.current_patches:
                top_patch = self.current_patches[0]
                apply_result = self.patch_applier.apply(top_patch.patch)
                
                if apply_result.success:
                    logger.info("Patch applied successfully")
                    return ExecResult(
                        status="ok",
                        summary=f"Generated and applied {len(result.candidates)} patches",
                        metrics={
                            "patches_generated": len(result.candidates),
                            "patch_applied": True,
                            "tokens_used": result.tokens_used,
                        },
                    )
                else:
                    logger.warning(f"Patch application failed: {apply_result.error}")
                    return ExecResult(
                        status="fail",
                        summary=f"Patch application failed: {apply_result.error}",
                        metrics={
                            "patches_generated": len(result.candidates),
                            "patch_applied": False,
                        },
                    )
        
        return ExecResult(
            status="fail",
            summary="No patches generated",
            metrics={"patches_generated": 0},
        )
    
    def _exec_staged_tests(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute staged test validation."""
        logger.info("Running staged tests")
        
        test_command = proposal.inputs.get("test_command", self.test_command)
        use_docker = proposal.inputs.get("use_docker", self.use_docker)
        
        # Run staged tests
        config = TestStageConfig(
            repo_path=str(self.repo_path),
            test_command=test_command,
            use_docker=use_docker,
        )
        
        result = run_staged_tests(config)
        
        # Analyze results
        passed = result.validation_passed and not result.new_regressions
        
        if passed:
            logger.info("Staged tests PASSED")
            return ExecResult(
                status="ok",
                summary=f"Staged tests passed (fixed={result.fixed_regressions}, no new regressions)",
                metrics={
                    "test_result": {
                        "passed": True,
                        "fixed_regressions": result.fixed_regressions,
                        "new_regressions": result.new_regressions,
                        "duration": result.total_duration,
                    }
                },
            )
        else:
            logger.warning(f"Staged tests FAILED (new regressions: {result.new_regressions})")
            
            # Store failures for diagnosis
            state.last_failures = []
            
            return ExecResult(
                status="fail",
                summary=f"Staged tests failed (new regressions: {result.new_regressions})",
                metrics={
                    "test_result": {
                        "passed": False,
                        "new_regressions": result.new_regressions,
                        "duration": result.total_duration,
                    }
                },
            )
    
    def _exec_diagnose(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute failure diagnosis."""
        logger.info("Diagnosing test failures")
        
        failures = proposal.inputs.get("failures", [])
        
        if not failures:
            return ExecResult(
                status="ok",
                summary="No failures to diagnose",
            )
        
        # Triage each failure
        diagnoses = []
        for failure in failures:
            diagnosis = triage_failure(
                test_id=failure.nodeid,
                error_message=failure.message,
                traceback=failure.traceback or "",
                repo_path=str(self.repo_path),
            )
            diagnoses.append(diagnosis)
        
        # Store diagnoses
        state.notes["failure_diagnoses"] = [
            {
                "failure_type": d.failure_type.value,
                "severity": d.severity.value,
                "root_cause": d.root_cause,
                "suggested_fix": d.suggested_fix,
                "confidence": d.confidence,
            }
            for d in diagnoses
        ]
        
        logger.info(f"Diagnosed {len(diagnoses)} failures")
        
        return ExecResult(
            status="ok",
            summary=f"Diagnosed {len(diagnoses)} failures",
            metrics={"diagnoses": len(diagnoses)},
        )
    
    def _exec_minimize_patch(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute patch minimization."""
        logger.info("Minimizing patch")
        
        if not self.current_patches:
            return ExecResult(
                status="fail",
                summary="No patch to minimize",
            )
        
        # Get current best patch
        current_patch = self.current_patches[0]
        
        # Minimize
        minimized = self.patch_minimizer.minimize(
            patch=current_patch.patch,
            test_command=self.test_command,
            repo_path=str(self.repo_path),
        )
        
        if minimized:
            logger.info(f"Patch minimized: {len(current_patch.patch.diffs)} -> {len(minimized.diffs)} diffs")
            
            # Update current patch
            current_patch.patch = minimized
            
            return ExecResult(
                status="ok",
                summary=f"Patch minimized to {len(minimized.diffs)} diffs",
                metrics={"minimized_diffs": len(minimized.diffs)},
            )
        else:
            logger.warning("Patch minimization failed")
            return ExecResult(
                status="fail",
                summary="Patch minimization failed",
            )
    
    def _exec_finalize(self, state: AgentState, proposal: Proposal) -> ExecResult:
        """Execute finalization."""
        logger.info("Finalizing episode")
        
        create_pr = proposal.inputs.get("create_pr", False)
        commit_changes = proposal.inputs.get("commit_changes", True)
        
        # For now, just mark as complete
        # Real finalization would create PR/commit
        
        return ExecResult(
            status="ok",
            summary="Episode finalized",
            metrics={
                "create_pr": create_pr,
                "commit_changes": commit_changes,
            },
        )


def run_orchestrated_episode(
    task_id: str,
    problem_statement: str,
    repo_path: str,
    test_command: str = "pytest tests/",
    profile_name: str = "swebench_lite",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4-turbo-preview",
    use_docker: bool = False,
) -> AgentState:
    """Run a complete orchestrated episode (convenience function).
    
    Args:
        task_id: Task identifier
        problem_statement: Problem description
        repo_path: Path to repository
        test_command: Test command
        profile_name: Profile name (swebench_lite, swebench_verified)
        llm_provider: LLM provider (openai, anthropic, deepseek)
        llm_model: LLM model name
        use_docker: Use Docker for test isolation
        
    Returns:
        Final agent state
        
    Example:
        >>> final_state = run_orchestrated_episode(
        ...     task_id="django-12345",
        ...     problem_statement="Fix AttributeError in ...",
        ...     repo_path="/tmp/django",
        ...     test_command="pytest tests/test_models.py",
        ... )
    """
    from .profiles import load_profile
    from .types import AgentState, RepoFingerprint, BudgetState, Phase
    
    # Load profile
    profile = load_profile(f"profiles/{profile_name}.yaml")
    
    # Initialize state
    repo = RepoFingerprint(
        repo_id=task_id,
        commit_sha="HEAD",
        workdir=repo_path,
        language="python",
    )
    
    budget = BudgetState(
        max_rounds=profile.max_rounds,
    )
    
    state = AgentState(
        task_id=task_id,
        repo=repo,
        phase=Phase.INGEST,
        budget=budget,
        notes={"problem_statement": problem_statement},
    )
    
    # Create orchestrator
    orchestrator = EpisodeOrchestrator(
        profile=profile,
        repo_path=repo_path,
        test_command=test_command,
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_docker=use_docker,
    )
    
    # Run episode
    final_state = orchestrator.run(state)
    
    return final_state
