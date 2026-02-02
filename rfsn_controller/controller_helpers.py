"""Controller helper functions.

Extracted from controller.py for better modularity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Forbidden path prefixes that should not be sent to the model
FORBIDDEN_PREFIXES = [".git/", "node_modules/", ".venv/", "venv/", "__pycache__/"]


def truncate(s: str, limit: int) -> str:
    """Truncate string to limit."""
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def diff_hash(d: str) -> str:
    """Compute a hash of a diff string for deduplication."""
    return hashlib.sha256(d.encode()).hexdigest()[:16]


def safe_path(p: str) -> bool:
    """Return True if the relative path is outside forbidden prefixes."""
    return not any(p.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def files_block(files: list[dict[str, Any]]) -> str:
    """Create a files block for the model input from a list of read_file results."""
    parts = []
    for f in files:
        path = f.get("path", "unknown")
        content = f.get("content", "")
        parts.append(f"--- {path} ---\n{content}\n")
    return "\n".join(parts)


def constraints_text() -> str:
    """Return a static constraints description for the model."""
    return """
CONSTRAINTS:
- Only modify files that are relevant to fixing the failing tests
- Do not modify test files unless they contain bugs
- Preserve existing functionality while fixing issues
- Use minimal, targeted changes
- Follow the existing code style and conventions
- Do not add new dependencies unless absolutely necessary
- Ensure all modified code is syntactically correct
"""


def infer_buildpack_type_from_test_cmd(test_cmd: str) -> str:
    """Infer buildpack type from test command.
    
    Args:
        test_cmd: The test command string.
        
    Returns:
        Inferred buildpack type string.
    """
    test_cmd_lower = test_cmd.lower()
    
    if any(x in test_cmd_lower for x in ["pytest", "python", "pip", "unittest"]):
        return "python"
    elif any(x in test_cmd_lower for x in ["npm", "yarn", "jest", "mocha", "node"]):
        return "node"
    elif any(x in test_cmd_lower for x in ["cargo", "rustc"]):
        return "rust"
    elif any(x in test_cmd_lower for x in ["go test", "go build"]):
        return "go"
    elif any(x in test_cmd_lower for x in ["mvn", "gradle", "java"]):
        return "java"
    elif any(x in test_cmd_lower for x in ["dotnet", "csharp"]):
        return "dotnet"
    elif any(x in test_cmd_lower for x in ["bundle", "rake", "rspec", "ruby"]):
        return "ruby"
    else:
        return "unknown"


def extract_traceback_files(traceback_text: str) -> list[str]:
    """Extract file paths mentioned in a traceback.
    
    Args:
        traceback_text: The traceback text to parse.
        
    Returns:
        List of file paths found in the traceback.
    """
    # Match patterns like: File "/path/to/file.py", line N
    pattern = r'File "([^"]+)", line \d+'
    matches = re.findall(pattern, traceback_text)
    return list(set(matches))


def normalize_file_path(path: str, repo_dir: str) -> str:
    """Normalize a file path to be relative to the repo directory.
    
    Args:
        path: Absolute or relative file path.
        repo_dir: The repository root directory.
        
    Returns:
        Normalized relative path.
    """
    import os
    
    if os.path.isabs(path):
        if path.startswith(repo_dir):
            return os.path.relpath(path, repo_dir)
    return path
