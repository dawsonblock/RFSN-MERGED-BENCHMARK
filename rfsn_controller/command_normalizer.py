"""Command normalization and shell idiom detection.
from __future__ import annotations

This module provides functions to detect shell idioms in commands and
prevent the model from wasting quota on commands that cannot execute
due to shell=False execution model.
"""

import re
import shlex


def detect_shell_idioms(cmd: str) -> bool:
    """Detect if a command contains shell idioms that won't work with shell=False.

    Shell idioms include:
    - Command chaining: &&, ||, ;
    - Pipes: |
    - Redirects: >, <, >>
    - Command substitution: $(, backticks
    - Newlines/carriage returns for multi-line commands
    - cd commands (since commands run from repo root)
    - Inline environment variables: VAR=value command

    Args:
        cmd: The command string to check.

    Returns:
        True if shell idioms are detected, False otherwise.
    """
    # Check for command chaining operators
    if "&&" in cmd or "||" in cmd or ";" in cmd:
        return True

    # Check for pipes and redirects
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = []

    # If tokenization worked, look for actual operator tokens (not characters inside quotes)
    if tokens:
        if any(t in {"|", ">", "<", ">>"} for t in tokens):
            return True
    elif "|" in cmd or ">" in cmd or "<" in cmd:
        return True

    # Check for command substitution
    if "$(" in cmd or "`" in cmd:
        return True

    # Check for newlines (multi-line commands)
    if "\n" in cmd or "\r" in cmd:
        return True

    # Check for cd command (commands run from repo root)
    cmd_lower = cmd.strip().lower()
    if cmd_lower.startswith("cd ") or " cd " in cmd_lower:
        return True

    # Check for inline environment variables: VAR=value command
    # Pattern: word=value at start of command before actual command
    env_var_pattern = r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+"
    if re.match(env_var_pattern, cmd):
        return True

    return False


def split_compound(cmd: str) -> list[str]:
    """Split a compound command into individual commands.

    This is a conservative splitter that only handles simple cases:
    - Splits on && (AND operator)
    - Strips whitespace

    If no compound operators are found, returns the original command as a single-item list.

    Note: This is intentionally conservative. Complex shell commands should be
    rejected rather than attempting to auto-split them.

    Args:
        cmd: The command string to split.

    Returns:
        List of individual command strings.
    """
    # Only split on && for now (most common case)
    if "&&" in cmd:
        parts = cmd.split("&&")
        return [part.strip() for part in parts if part.strip()]

    # Return original command as single item
    return [cmd.strip()] if cmd.strip() else []


def get_shell_idiom_error_message(cmd: str) -> str:
    """Generate a helpful error message for shell idiom detection.

    Args:
        cmd: The command that contains shell idioms.

    Returns:
        Error message explaining the issue and how to fix it.
    """
    issues = []

    if "&&" in cmd or "||" in cmd or ";" in cmd:
        issues.append("command chaining (&&, ||, ;)")
    if "|" in cmd:
        issues.append("pipes (|)")
    if ">" in cmd or "<" in cmd:
        issues.append("redirects (>, <)")
    if "$(" in cmd or "`" in cmd:
        issues.append("command substitution")
    if "\n" in cmd or "\r" in cmd:
        issues.append("multi-line commands")
    if " cd " in cmd.lower() or cmd.lower().startswith("cd "):
        issues.append("cd command")
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+", cmd):
        issues.append("inline environment variables")

    issue_list = ", ".join(issues)

    return (
        f"Shell syntax detected: {issue_list}. "
        "Commands are executed with shell=False. "
        "Split into separate tool requests or use simple commands only. "
        "Commands run from repo root - cd is not needed."
    )
