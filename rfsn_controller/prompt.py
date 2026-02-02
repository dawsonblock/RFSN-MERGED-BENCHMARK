"""Helpers for constructing model input strings."""
from __future__ import annotations

from typing import Any

# Mode constants
MODE_FEATURE = "feature"


def _truncate(s: str, n: int) -> str:
    """Truncate a string to at most n characters, appending a marker if truncated.

    Args:
        s: String to truncate
        n: Maximum length

    Returns:
        Truncated string or empty string if input is empty
    """
    if not s:
        return ""
    # Optimize: avoid string concatenation if not needed
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]..."


def build_model_input(state: dict[str, Any]) -> str:
    """Build a formatted model input string from the controller state.

    The state dictionary should contain keys: goal, intent, subgoal, test_cmd,
    focus_test_cmd, failure_output, repo_tree, constraints, files_block, observations.

    For feature mode, additional keys: feature_description, acceptance_criteria,
    current_subgoal_index, completed_subgoals.

    Args:
        state: A mapping of context fields to embed in the prompt.

    Returns:
        A single string with sections separated by headers.

    Raises:
        KeyError: If required state keys are missing
    """
    # Validate required keys
    required_keys = [
        "goal",
        "test_cmd",
        "focus_test_cmd",
        "failure_output",
        "repo_tree",
        "constraints",
        "files_block",
    ]
    missing_keys = [k for k in required_keys if k not in state]
    if missing_keys:
        raise KeyError(f"Missing required state keys: {missing_keys}")

    # Check if this is feature mode
    is_feature_mode = state.get("mode") == MODE_FEATURE

    # Use list for efficient string building
    sections = [f"GOAL:\n{state['goal']}\n\n"]

    if is_feature_mode:
        # Feature mode sections
        feature_desc = state.get("feature_description", "")
        if feature_desc:
            sections.append(f"FEATURE_DESCRIPTION:\n{feature_desc}\n\n")

        acceptance_criteria = state.get("acceptance_criteria", [])
        if acceptance_criteria:
            # Optimize: use join instead of repeated concatenation
            criteria_text = "\n".join(f"  - {c}" for c in acceptance_criteria)
            sections.append(f"ACCEPTANCE_CRITERIA:\n{criteria_text}\n\n")

        completed_subgoals = state.get("completed_subgoals", [])
        if completed_subgoals:
            # Optimize: use join instead of repeated concatenation
            completed_text = "\n".join(f"  ✓ {s}" for s in completed_subgoals)
            sections.append(f"COMPLETED_SUBGOALS:\n{completed_text}\n\n")

        current_subgoal = state.get("current_subgoal", "")
        if current_subgoal:
            sections.append(f"CURRENT_SUBGOAL:\n{current_subgoal}\n\n")
    else:
        # Repair mode sections (original behavior)
        # Validate repair mode required keys
        if "intent" not in state or "subgoal" not in state:
            raise KeyError("Repair mode requires 'intent' and 'subgoal' keys")
        sections.extend(
            [
                f"INTENT:\n{state['intent']}\n\n",
                f"SUBGOAL:\n{state['subgoal']}\n\n",
            ]
        )

    # Add common sections
    sections.extend(
        [
            f"TEST_COMMAND:\n{state['test_cmd']}\n\n",
            f"FOCUS_TEST_COMMAND:\n{state['focus_test_cmd']}\n\n",
            f"FAILURE_OUTPUT:\n{_truncate(state['failure_output'], 45000)}\n\n",
            f"REPO_TREE:\n{_truncate(state['repo_tree'], 20000)}\n\n",
            f"RETRIEVAL_CONTEXT:\n{_truncate(state.get('retrieval_context',''), 20000)}\n\n"
            if state.get("retrieval_context")
            else "",
            f"CONSTRAINTS:\n{state['constraints']}\n\n",
            f"FILES:\n{state['files_block']}\n",
            "\nIMPORTANT: Do not loop. If you have already read a file, do not read it again unless you changed it. If you have gathered enough evidence, propose a patch immediately.\n",
        ]
    )

    # Add optional sections
    action_priors = state.get("action_priors")
    if action_priors:
        sections.append(f"\nACTION_PRIORS:\n{_truncate(action_priors, 12000)}\n")

    observations = state.get("observations")
    if observations:
        sections.append(f"\nOBSERVATIONS:\n{_truncate(observations, 30000)}")

    # Use join for efficient string concatenation
    return "".join(sections)
