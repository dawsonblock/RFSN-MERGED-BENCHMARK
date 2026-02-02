"""Tests for the CGW-based coding agent.

This test module validates the serial decision architecture:
- Single commit per cycle (seriality guarantee)
- Forced signal bypass (safety overrides)
- Blocking execution (no tool overlap)
- Event emission for replay
- Deterministic decision replay
"""

import time

import pytest

from cgw_ssl_guard import SimpleEventBus, ThalamusGate
from cgw_ssl_guard.coding_agent import (
    ActionPayload,
    AgentConfig,
    AgentResult,
    BlockingExecutor,
    CodingAction,
    CodingAgentRuntime,
    ExecutionResult,
    ExecutorConfig,
    IdleProposalGenerator,
    PlannerProposalGenerator,
    ProposalContext,
    SafetyProposalGenerator,
)
from cgw_ssl_guard.monitors import SerialityMonitor
from cgw_ssl_guard.types import Candidate


class TestSerialityGuarantee:
    """Tests for the single-commit-per-cycle invariant."""
    
    def test_single_commit_per_cycle(self):
        """Verify SerialityMonitor shows exactly 1 commit per cycle."""
        event_bus = SimpleEventBus()
        monitor = SerialityMonitor()
        event_bus.on("CGW_COMMIT", monitor.on_commit)
        
        runtime = CodingAgentRuntime(
            config=AgentConfig(max_cycles=5),
            event_bus=event_bus,
        )
        
        # Run a few cycles
        for _ in range(3):
            runtime.tick()
        
        # Verify each cycle had exactly one commit
        for cycle_id, count in monitor.commits_per_cycle.items():
            assert count == 1, f"Cycle {cycle_id} had {count} commits, expected 1"
    
    def test_verify_seriality_method(self):
        """Test the runtime's seriality verification method."""
        runtime = CodingAgentRuntime(config=AgentConfig(max_cycles=10))
        
        # Run some cycles
        for _ in range(5):
            runtime.tick()
        
        # Seriality should be maintained
        assert runtime.verify_seriality() is True


class TestForcedSignalOverride:
    """Tests for forced signal bypassing competition."""
    
    def test_forced_signal_wins_regardless_of_score(self):
        """A forced signal should win over any normal candidates."""
        event_bus = SimpleEventBus()
        gate = ThalamusGate(event_bus)
        
        # Submit a high-scoring normal candidate
        normal = Candidate(
            slot_id="normal_high",
            source_module="planner",
            content_payload=ActionPayload(action=CodingAction.RUN_TESTS).to_bytes(),
            saliency=1.0,
            urgency=1.0,
            surprise=1.0,
        )
        gate.submit_candidate(normal)
        
        # Inject a forced signal
        gate.inject_forced_signal(
            source_module="safety",
            content_payload=ActionPayload(action=CodingAction.ABORT).to_bytes(),
            reason="emergency",
        )
        
        # The forced signal should win
        winner, reason = gate.select_winner()
        assert winner is not None
        payload = ActionPayload.from_bytes(winner.content_payload)
        assert payload.action == CodingAction.ABORT
    
    def test_inject_abort_stops_execution(self):
        """Injecting an abort signal should terminate the run loop."""
        runtime = CodingAgentRuntime(config=AgentConfig(max_cycles=100))
        
        # Inject abort before running
        runtime.inject_forced_signal(CodingAction.ABORT, "test_abort")
        
        # Run should stop immediately
        result = runtime.run_until_done()
        
        assert result.final_action == CodingAction.ABORT
        assert result.cycles_executed == 1
        assert result.success is False


class TestBlockingExecution:
    """Tests for blocking execution guarantee."""
    
    def test_executor_blocks_concurrent_execution(self):
        """Executor should raise if called while already executing."""
        executor = BlockingExecutor(config=ExecutorConfig())
        
        # Simulate execution in progress
        executor._is_executing = True
        
        payload = ActionPayload(action=CodingAction.RUN_TESTS)
        
        with pytest.raises(RuntimeError, match="Cannot execute"):
            executor.execute(payload)
    
    def test_is_executing_flag(self):
        """The is_executing flag should track execution state."""
        executor = BlockingExecutor(config=ExecutorConfig())
        
        assert executor.is_executing() is False
        
        payload = ActionPayload(action=CodingAction.IDLE)
        executor.execute(payload)
        
        # After execution, flag should be False
        assert executor.is_executing() is False
    
    def test_execution_count_tracking(self):
        """Executor should track the number of executions."""
        executor = BlockingExecutor(config=ExecutorConfig())
        
        assert executor.execution_count() == 0
        
        for i in range(5):
            executor.execute(ActionPayload(action=CodingAction.IDLE))
            assert executor.execution_count() == i + 1


class TestEventEmission:
    """Tests for event emission for replay and auditing."""
    
    def test_gate_selection_event_emitted(self):
        """GATE_SELECTION event should be emitted when winner selected."""
        events = []
        event_bus = SimpleEventBus()
        event_bus.on("GATE_SELECTION", lambda e: events.append(e))
        
        gate = ThalamusGate(event_bus)
        candidate = Candidate(
            slot_id="test_candidate",
            source_module="test",
            content_payload=ActionPayload(action=CodingAction.RUN_TESTS).to_bytes(),
            saliency=0.5,
        )
        gate.submit_candidate(candidate)
        gate.select_winner()
        
        assert len(events) == 1
        assert events[0].slot_id == "test_candidate"
    
    def test_cgw_commit_event_emitted(self):
        """CGW_COMMIT event should be emitted when state committed."""
        events = []
        event_bus = SimpleEventBus()
        event_bus.on("CGW_COMMIT", lambda e: events.append(e))
        
        runtime = CodingAgentRuntime(
            config=AgentConfig(max_cycles=3),
            event_bus=event_bus,
        )
        
        runtime.tick()
        
        assert len(events) >= 1
        assert "cycle_id" in events[0]
        assert "slot_id" in events[0]
    
    def test_forced_injection_event(self):
        """FORCED_INJECTION event should be emitted when forced signal added."""
        events = []
        event_bus = SimpleEventBus()
        event_bus.on("FORCED_INJECTION", lambda e: events.append(e))
        
        gate = ThalamusGate(event_bus)
        gate.inject_forced_signal(
            source_module="test",
            content_payload=b"test",
            reason="test_reason",
        )
        
        assert len(events) == 1
        assert events[0]["source"] == "test"


class TestCodingWorkflow:
    """Tests for the full coding workflow sequence."""
    
    def test_initial_action_is_run_tests(self):
        """First cycle should propose running tests."""
        runtime = CodingAgentRuntime(config=AgentConfig())
        result = runtime.tick()
        
        assert result.action == CodingAction.RUN_TESTS
    
    def test_workflow_reaches_finalize(self):
        """A successful workflow should reach FINALIZE."""
        # Create a runtime with mocked passing tests
        runtime = CodingAgentRuntime(config=AgentConfig(max_cycles=10))
        
        # Manually set context to simulate passing tests
        runtime._context.tests_passing = True
        runtime._context.last_action = CodingAction.RUN_TESTS
        
        result = runtime.tick()
        
        # Should propose finalize
        assert result.action == CodingAction.FINALIZE
    
    def test_cycle_history_recorded(self):
        """All cycles should be recorded in history."""
        runtime = CodingAgentRuntime(config=AgentConfig(max_cycles=5))
        
        # Run at least one cycle
        runtime.tick()
        
        history = runtime.get_cycle_history()
        
        # Verify at least one cycle was recorded
        assert len(history) >= 1
        
        # Verify cycle ID is correct
        assert history[0].cycle_id == 1
        
        # Verify the first action was RUN_TESTS
        assert history[0].action == CodingAction.RUN_TESTS


class TestDeterministicReplay:
    """Tests for deterministic decision replay."""
    
    def test_same_candidates_same_winner(self):
        """Given the same candidates, the gate should select the same winner."""
        event_bus1 = SimpleEventBus()
        event_bus2 = SimpleEventBus()
        gate1 = ThalamusGate(event_bus1)
        gate2 = ThalamusGate(event_bus2)
        
        # Create identical candidates
        candidates = [
            Candidate(
                slot_id="candidate_a",
                source_module="test",
                content_payload=b"action_a",
                saliency=0.5,
                urgency=0.3,
            ),
            Candidate(
                slot_id="candidate_b",
                source_module="test",
                content_payload=b"action_b",
                saliency=0.8,  # Higher score
                urgency=0.1,
            ),
        ]
        
        # Submit to both gates
        for c in candidates:
            gate1.submit_candidate(c)
        for c in candidates:
            gate2.submit_candidate(c)
        
        # Wait for cooldown
        time.sleep(0.15)
        
        # Both should select the same winner
        winner1, _ = gate1.select_winner()
        winner2, _ = gate2.select_winner()
        
        assert winner1.slot_id == winner2.slot_id


class TestProposalGenerators:
    """Tests for proposal generators."""
    
    def test_safety_generator_on_trigger(self):
        """SafetyProposalGenerator should propose ABORT when triggered."""
        generator = SafetyProposalGenerator()
        context = ProposalContext(
            cycle_id=1,
            safety_triggered=True,
            safety_reason="max_patches_exceeded",
        )
        
        candidates = generator.generate(context)
        
        assert len(candidates) == 1
        payload = ActionPayload.from_bytes(candidates[0].content_payload)
        assert payload.action == CodingAction.ABORT
    
    def test_safety_generator_no_trigger(self):
        """SafetyProposalGenerator should return empty when not triggered."""
        generator = SafetyProposalGenerator()
        context = ProposalContext(cycle_id=1, safety_triggered=False)
        
        candidates = generator.generate(context)
        
        assert len(candidates) == 0
    
    def test_idle_generator_always_proposes(self):
        """IdleProposalGenerator should always propose IDLE."""
        generator = IdleProposalGenerator()
        context = ProposalContext(cycle_id=1)
        
        candidates = generator.generate(context)
        
        assert len(candidates) == 1
        payload = ActionPayload.from_bytes(candidates[0].content_payload)
        assert payload.action == CodingAction.IDLE
    
    def test_planner_generator_heuristic_sequence(self):
        """PlannerProposalGenerator should follow heuristic sequence."""
        generator = PlannerProposalGenerator()
        
        # Test initial state -> RUN_TESTS
        context = ProposalContext(cycle_id=1)
        candidates = generator.generate(context)
        assert len(candidates) > 0
        payload = ActionPayload.from_bytes(candidates[0].content_payload)
        assert payload.action == CodingAction.RUN_TESTS
        
        # Test after failed tests -> ANALYZE_FAILURE
        context = ProposalContext(
            cycle_id=2,
            last_action=CodingAction.RUN_TESTS,
            tests_passing=False,
            failing_tests=["test_foo.py"],
        )
        candidates = generator.generate(context)
        assert len(candidates) > 0
        payload = ActionPayload.from_bytes(candidates[0].content_payload)
        assert payload.action == CodingAction.ANALYZE_FAILURE


class TestAgentResult:
    """Tests for agent result formatting."""
    
    def test_result_summary(self):
        """AgentResult should generate a readable summary."""
        result = AgentResult(
            success=True,
            final_action=CodingAction.FINALIZE,
            cycles_executed=5,
            total_time_ms=1234.5,
            tests_passing=True,
            patches_applied=2,
        )
        
        summary = result.summary()
        
        assert "[SUCCESS]" in summary
        assert "FINALIZE" in summary
        assert "5 cycles" in summary
        assert "Tests passing: True" in summary
    
    def test_result_failure_summary(self):
        """Failed result should indicate failure."""
        result = AgentResult(
            success=False,
            final_action=CodingAction.ABORT,
            cycles_executed=10,
            total_time_ms=5000.0,
            error="max_cycles_exceeded",
        )
        
        summary = result.summary()
        
        assert "[FAILURE]" in summary
        assert "ABORT" in summary


class TestActionPayload:
    """Tests for action payload serialization."""
    
    def test_payload_roundtrip(self):
        """ActionPayload should serialize and deserialize correctly."""
        original = ActionPayload(
            action=CodingAction.APPLY_PATCH,
            parameters={"diff": "--- a/file.py\n+++ b/file.py"},
            context={"reason": "fix_bug"},
            metadata={"generator": "planner"},
        )
        
        serialized = original.to_bytes()
        restored = ActionPayload.from_bytes(serialized)
        
        assert restored.action == original.action
        assert restored.parameters == original.parameters
        assert restored.context == original.context
        assert restored.metadata == original.metadata


class TestExecutionResult:
    """Tests for execution result handling."""
    
    def test_terminal_detection(self):
        """ExecutionResult should detect terminal actions."""
        finalize = ExecutionResult(action=CodingAction.FINALIZE, success=True)
        abort = ExecutionResult(action=CodingAction.ABORT, success=True)
        run_tests = ExecutionResult(action=CodingAction.RUN_TESTS, success=True)
        
        assert finalize.is_terminal() is True
        assert abort.is_terminal() is True
        assert run_tests.is_terminal() is False
