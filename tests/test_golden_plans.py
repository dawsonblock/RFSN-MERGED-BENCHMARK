"""Golden Plan Test Runner.

Tests that generated plans match expected invariants and patterns.
Prevents planner changes from silently degrading plan quality.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from rfsn_controller.planner_v2 import ControllerAdapter

GOLDEN_PLANS_DIR = Path(__file__).parent / "golden_plans"


def load_golden_examples() -> list[dict[str, Any]]:
    """Load all golden plan examples."""
    examples = []
    for path in GOLDEN_PLANS_DIR.glob("*.json"):
        with open(path) as f:
            example = json.load(f)
            example["_path"] = str(path)
            examples.append(example)
    return examples


def get_example_ids() -> list[str]:
    """Get IDs for parametrized tests."""
    examples = load_golden_examples()
    return [e["name"] for e in examples]


class TestGoldenPlanInvariants:
    """Test that generated plans match golden example invariants."""
    
    @pytest.fixture
    def adapter(self) -> ControllerAdapter:
        """Create a controller adapter for testing."""
        return ControllerAdapter(seed=42)
    
    @pytest.mark.parametrize(
        "example",
        load_golden_examples(),
        ids=get_example_ids(),
    )
    def test_plan_meets_invariants(self, adapter: ControllerAdapter, example: dict[str, Any]):
        """Test that generated plan meets golden example invariants."""
        # Generate plan
        goal = example["goal"]
        context = example.get("context", {})
        
        # Start goal to generate plan
        adapter.start_goal(goal, context, validate=False)
        plan = adapter.get_plan()
        
        assert plan is not None, "Plan should be generated"
        
        invariants = example.get("invariants", {})
        
        # Check step count
        if "min_steps" in invariants:
            assert len(plan.steps) >= invariants["min_steps"], \
                f"Plan has {len(plan.steps)} steps, expected min {invariants['min_steps']}"
        
        if "max_steps" in invariants:
            assert len(plan.steps) <= invariants["max_steps"], \
                f"Plan has {len(plan.steps)} steps, expected max {invariants['max_steps']}"
        
        # Check forbidden files
        if "forbidden_files" in invariants:
            for step in plan.steps:
                for pattern in invariants["forbidden_files"]:
                    for allowed_file in step.allowed_files:
                        # Simple glob matching
                        regex = pattern.replace("*", ".*")
                        assert not re.match(regex, allowed_file), \
                            f"Step {step.step_id} allows forbidden file pattern {pattern}: {allowed_file}"
        
        # Check verification requirement
        if invariants.get("must_have_verify", False):
            has_verify = any(s.verify for s in plan.steps)
            assert has_verify, "Plan must have at least one step with verify command"
    
    @pytest.mark.parametrize(
        "example",
        load_golden_examples(),
        ids=get_example_ids(),
    )
    def test_plan_has_required_step_types(self, adapter: ControllerAdapter, example: dict[str, Any]):
        """Test that plan has required step type patterns."""
        goal = example["goal"]
        context = example.get("context", {})
        
        adapter.start_goal(goal, context, validate=False)
        plan = adapter.get_plan()
        
        invariants = example.get("invariants", {})
        required_types = invariants.get("required_step_types", [])
        
        step_ids_lower = [s.step_id.lower() for s in plan.steps]
        step_intents_lower = [s.intent.lower() for s in plan.steps]
        
        for required in required_types:
            found = any(
                required in step_id or required in intent
                for step_id, intent in zip(step_ids_lower, step_intents_lower, strict=True)
            )
            assert found, f"Plan missing required step type: {required}"
    
    @pytest.mark.parametrize(
        "example",
        load_golden_examples(),
        ids=get_example_ids(),
    )
    def test_plan_step_patterns(self, adapter: ControllerAdapter, example: dict[str, Any]):
        """Test that plan steps match expected patterns."""
        goal = example["goal"]
        context = example.get("context", {})
        
        adapter.start_goal(goal, context, validate=False)
        plan = adapter.get_plan()
        
        patterns = example.get("expected_step_pattern", [])
        
        for pattern_spec in patterns:
            if pattern_spec.get("optional", False):
                continue  # Skip optional patterns
            
            pattern = pattern_spec["pattern"]
            regex = re.compile(pattern, re.IGNORECASE)
            
            found = any(
                regex.search(s.step_id) or regex.search(s.intent) or regex.search(s.title)
                for s in plan.steps
            )
            
            assert found, \
                f"Plan missing step matching pattern '{pattern}': {pattern_spec.get('description', '')}"
