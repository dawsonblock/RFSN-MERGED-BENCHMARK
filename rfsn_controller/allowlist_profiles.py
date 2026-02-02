"""Language-scoped command allowlists for secure sandbox execution.
from __future__ import annotations

This module provides language-specific command allowlists that are selected
based on project detection. This prevents "global allowlist creep" where
adding support for one language makes its tools available to all projects.
"""

from typing import Any

# Base commands available to all projects (safe unix + git + common utilities)
BASE_COMMANDS: set[str] = {
    # Version control
    "git",
    # Unix utilities (safe subset)
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "ls",
    "pwd",
    "echo",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "touch",
    "chmod",
    "sed",
    "awk",
    "sort",
    "uniq",
    "wc",
    "diff",
    "patch",
    "tar",
    "unzip",
    "make",
}


# Python-specific commands
PYTHON_COMMANDS: set[str] = {
    "python",
    "python3",
    "pip",
    "pip3",
    "pytest",
    "pipenv",
    "poetry",
    "ruff",
    "mypy",
    "black",
    "flake8",
    "pylint",
    "tox",
    "coverage",
    "sphinx-build",
}


# Node.js / JavaScript / TypeScript commands
NODE_COMMANDS: set[str] = {
    "node",
    "npm",
    "yarn",
    "pnpm",
    "npx",
    "bun",
    "tsc",
    "jest",
    "mocha",
    "eslint",
    "prettier",
    "webpack",
    "vite",
    "next",
}


# Rust commands
RUST_COMMANDS: set[str] = {
    "cargo",
    "rustc",
    "rustup",
    "rustfmt",
    "clippy",
}


# Go commands
GO_COMMANDS: set[str] = {
    "go",
    "gofmt",
    "golint",
    "goimports",
}


# Java commands
JAVA_COMMANDS: set[str] = {
    "mvn",
    "gradle",
    "javac",
    "java",
    "ant",
}


# .NET / C# commands
DOTNET_COMMANDS: set[str] = {
    "dotnet",
    "nuget",
    "msbuild",
}


def commands_for_language(language: str) -> set[str]:
    """Get the set of allowed commands for a specific language.

    Args:
        language: Language identifier (e.g., "python", "node", "rust", "go", "java", "dotnet")

    Returns:
        Set of allowed command names combining base commands and language-specific commands.
    """
    language_lower = language.lower() if language else ""

    # Map language identifiers to command sets
    language_commands: dict[str, set[str]] = {
        "python": PYTHON_COMMANDS,
        "py": PYTHON_COMMANDS,
        "node": NODE_COMMANDS,
        "nodejs": NODE_COMMANDS,
        "javascript": NODE_COMMANDS,
        "js": NODE_COMMANDS,
        "typescript": NODE_COMMANDS,
        "ts": NODE_COMMANDS,
        "rust": RUST_COMMANDS,
        "rs": RUST_COMMANDS,
        "go": GO_COMMANDS,
        "golang": GO_COMMANDS,
        "java": JAVA_COMMANDS,
        "dotnet": DOTNET_COMMANDS,
        "csharp": DOTNET_COMMANDS,
        "c#": DOTNET_COMMANDS,
        "cs": DOTNET_COMMANDS,
    }

    # Get language-specific commands (default to Python for backward compatibility)
    lang_cmds = language_commands.get(language_lower, PYTHON_COMMANDS)

    # Combine base commands with language-specific commands
    return BASE_COMMANDS | lang_cmds


def commands_for_project(project_info: Any) -> set[str]:
    """Get the set of allowed commands for a project based on detection results.

    This function accepts either a dict or an object with language/project_type fields.
    It safely extracts the language and returns the appropriate command set.

    Args:
        project_info: Project detection result (dict or object with language/project_type)

    Returns:
        Set of allowed command names for the project.
    """
    language = None

    # Try to extract language from various formats
    if isinstance(project_info, dict):
        # Dict format: check for 'language', 'project_type', or 'buildpack_type'
        language = (
            project_info.get("language")
            or project_info.get("project_type")
            or project_info.get("buildpack_type")
        )
    elif hasattr(project_info, "language"):
        language = project_info.language
    elif hasattr(project_info, "project_type"):
        language = project_info.project_type
    elif hasattr(project_info, "buildpack_type"):
        language = project_info.buildpack_type

    # Handle enum types
    if language and hasattr(language, "value"):
        language = language.value
    elif language and hasattr(language, "name"):
        language = language.name

    # Convert to string
    if language:
        language = str(language)

    # Default to Python if no language detected (backward compatibility)
    if not language or language.lower() == "unknown":
        language = "python"

    return commands_for_language(language)
