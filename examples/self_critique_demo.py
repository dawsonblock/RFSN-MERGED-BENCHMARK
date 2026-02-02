"""
Example: Self-Critique Validation

This example demonstrates how to use the self-critique system
to validate plans before submission to the safety gate.
"""

from rfsn_controller.gates.self_critique import (
    critique_plan,
    validate_plan_json,
    CritiqueResult,
)


def main():
    """Demonstrate self-critique validation."""
    
    # Example 1: Valid plan
    print("=" * 50)
    print("Example 1: Valid Plan")
    print("=" * 50)
    
    valid_plan = {
        "goal": "Fix bug in calculator module",
        "metadata": {"budget": 5},
        "expected_outcome": "All tests pass",
        "steps": [
            {
                "id": "read",
                "type": "read_file",
                "inputs": {"path": "src/calculator.py"},
                "expected_outcome": "File contents loaded",
            },
            {
                "id": "fix",
                "type": "apply_patch",
                "inputs": {"path": "src/calculator.py", "patch": "..."},
                "depends_on": ["read"],
                "expected_outcome": "Patch applied",
            },
            {
                "id": "verify",
                "type": "run_tests",
                "inputs": {"test_path": "tests/"},
                "depends_on": ["fix"],
                "expected_outcome": "All tests pass",
            },
        ],
    }
    
    report = critique_plan(valid_plan)
    print(f"Result: {report.result.value}")
    print(f"Hard failures: {len(report.hard_failures)}")
    print(f"Soft warnings: {len(report.soft_warnings)}")
    
    # Example 2: Plan with shell injection attempt
    print("\n" + "=" * 50)
    print("Example 2: Shell Injection Attempt")
    print("=" * 50)
    
    dangerous_plan = {
        "goal": "Run arbitrary command",
        "metadata": {"budget": 1},
        "steps": [
            {
                "id": "exploit",
                "type": "read_file",
                "inputs": {"command": "bash -c 'rm -rf /'"},
                "expected_outcome": "X",
            },
        ],
    }
    
    report = critique_plan(dangerous_plan)
    print(f"Result: {report.result.value}")
    print(f"Hard failures: {report.hard_failures}")
    
    # Example 3: Path traversal attempt
    print("\n" + "=" * 50)
    print("Example 3: Path Traversal Attempt")
    print("=" * 50)
    
    traversal_plan = {
        "goal": "Read sensitive file",
        "metadata": {"budget": 1},
        "steps": [
            {
                "id": "read",
                "type": "read_file",
                "inputs": {"path": "../../etc/passwd"},
                "expected_outcome": "X",
            },
        ],
    }
    
    report = critique_plan(traversal_plan)
    print(f"Result: {report.result.value}")
    print(f"Hard failures: {report.hard_failures}")
    
    # Example 4: JSON validation
    print("\n" + "=" * 50)
    print("Example 4: JSON Validation")
    print("=" * 50)
    
    plan_json = '''
    {
        "goal": "Test",
        "metadata": {"budget": 10},
        "expected_outcome": "Done",
        "steps": [
            {"id": "s1", "type": "read_file", "inputs": {"path": "test.py"}, "expected_outcome": "Read"}
        ]
    }
    '''
    
    report = validate_plan_json(plan_json)
    print(f"Result: {report.result.value}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Self-Critique Summary")
    print("=" * 50)
    print("✅ Valid plans → APPROVED")
    print("❌ Shell injection → REJECTED")
    print("❌ Path traversal → REJECTED")
    print("✅ Valid JSON → APPROVED")


if __name__ == "__main__":
    main()
