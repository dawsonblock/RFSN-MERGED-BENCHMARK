"""Pytest configuration and shared fixtures for RFSN Controller tests.

This module provides:
- Common fixtures for test isolation
- Test file generators for scanner tests
- Temporary directory management
- Mock utilities for subprocess testing
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# =============================================================================
# Path Configuration
# =============================================================================

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
RFSN_CONTROLLER_ROOT = PROJECT_ROOT / "rfsn_controller"

sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Pytest Configuration Hooks
# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    # Markers are defined in pyproject.toml but we can add dynamic ones here
    pass


def pytest_collection_modifyitems(
    config: pytest.Config, 
    items: list[pytest.Item]
) -> None:
    """Modify test collection - auto-apply markers based on path."""
    for item in items:
        # Auto-tag tests in unit/ directory
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Auto-tag security tests
        if "security" in item.name.lower() or "shell" in item.name.lower():
            item.add_marker(pytest.mark.security)


# =============================================================================
# Temporary File Fixtures
# =============================================================================

@pytest.fixture
def tmp_py_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture to create temporary Python files.
    
    Usage:
        def test_something(tmp_py_file):
            path = tmp_py_file("test.py", "print('hello')")
    """
    def _create(filename: str, content: str) -> Path:
        filepath = tmp_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        return filepath
    return _create


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Callable[[dict[str, str]], Path]:
    """Factory fixture to create a temporary project structure.
    
    Usage:
        def test_project(tmp_project_dir):
            project = tmp_project_dir({
                "main.py": "import os",
                "utils/helper.py": "def foo(): pass",
            })
    """
    def _create(files: dict[str, str]) -> Path:
        for rel_path, content in files.items():
            filepath = tmp_path / rel_path
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)
        return tmp_path
    return _create


# =============================================================================
# Shell Scanner Test Fixtures
# =============================================================================

@pytest.fixture
def safe_code_sample() -> str:
    """Return a sample of safe Python code without shell violations."""
    return '''
"""Safe module example."""
import subprocess
from typing import List

def run_safe(args: List[str]) -> subprocess.CompletedProcess:
    """Run a command safely with argv list."""
    return subprocess.run(
        args,
        shell=False,  # Explicitly safe
        capture_output=True,
        text=True,
    )

def main():
    result = run_safe(["echo", "hello"])
    print(result.stdout)
'''


@pytest.fixture
def unsafe_code_samples() -> dict[str, str]:
    """Return samples of unsafe Python code with various violations."""
    return {
        "shell_true": '''
import subprocess

def bad_run():
    subprocess.run("echo hello", shell=True)  # Violation!
''',
        "os_system": '''
import os

def bad_system():
    os.system("rm -rf /tmp/test")  # Violation!
''',
        "shell_wrapper": '''
import subprocess

def bad_wrapper():
    subprocess.run(["sh", "-c", "echo hello"], shell=False)  # Still bad!
''',
        "interactive_bash": '''
import subprocess

def spawn_shell():
    proc = subprocess.Popen(["/bin/bash", "-i"])  # Violation!
''',
        "popen_shell": '''
import os

def bad_popen():
    os.popen("ls -la")  # Violation!
''',
    }


@pytest.fixture
def mixed_code_project(tmp_path: Path) -> Path:
    """Create a temporary project with mixed safe/unsafe code."""
    files = {
        "safe_module.py": '''
import subprocess

def safe_run(cmd_list):
    return subprocess.run(cmd_list, shell=False, capture_output=True)
''',
        "unsafe_module.py": '''
import subprocess
import os

def unsafe_shell():
    subprocess.run("echo test", shell=True)

def unsafe_system():
    os.system("ls")
''',
        "subdir/nested.py": '''
import subprocess

def nested_violation():
    subprocess.call(["bash", "-c", "echo nested"])
''',
    }
    
    for rel_path, content in files.items():
        filepath = tmp_path / rel_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
    
    return tmp_path


# =============================================================================
# Exec Utils Test Fixtures
# =============================================================================

@pytest.fixture
def valid_argv_samples() -> list[list[str]]:
    """Return valid argv command lists for testing."""
    return [
        ["echo", "hello"],
        ["ls", "-la", "/tmp"],
        # Removed python -c: this is disallowed and should be treated as invalid
        ["git", "status"],
        ["docker", "ps", "-a"],
    ]


@pytest.fixture
def invalid_argv_samples() -> list[tuple]:
    """Return invalid argv samples with expected error types."""
    return [
        # (argv, expected_error_pattern)
        ("echo hello", "must be a list"),  # String instead of list
        ([], "must not be empty"),  # Empty list
        (["sh", "-c", "echo test"], "Shell wrapper"),  # Shell wrapper
        (["bash", "-c", "rm -rf /"], "Shell wrapper"),  # Shell wrapper
        (["/bin/bash", "-i"], "Shell wrapper"),  # Interactive shell - Fixed

        # Python exec-string now disallowed
        (["python", "-c", "print('hello')"], "Disallowed flag"),
    ]


@pytest.fixture
def test_cwd(tmp_path: Path) -> Path:
    """Provide a temporary working directory for exec tests."""
    return tmp_path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_env() -> dict[str, str]:
    """Provide a mock environment for testing."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp/testhome",
        "LANG": "en_US.UTF-8",
    }


@pytest.fixture
def capture_subprocess_calls(monkeypatch: pytest.MonkeyPatch):
    """Fixture to capture subprocess.run calls without executing them.
    
    Usage:
        def test_calls(capture_subprocess_calls):
            calls = capture_subprocess_calls
            # ... run code that uses subprocess ...
            assert len(calls) == 1
            assert calls[0]['args'] == ['echo', 'test']
    """
    calls: list[dict[str, Any]] = []
    
    def mock_run(*args, **kwargs):
        calls.append({
            'args': args[0] if args else kwargs.get('args'),
            'kwargs': kwargs,
        })
        # Return a mock CompletedProcess
        from subprocess import CompletedProcess
        return CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="mocked output",
            stderr="",
        )
    
    import subprocess
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    return calls


# =============================================================================
# Project Structure Fixtures
# =============================================================================

@pytest.fixture
def rfsn_controller_path() -> Path:
    """Return the path to the rfsn_controller package."""
    return RFSN_CONTROLLER_ROOT


@pytest.fixture
def project_root() -> Path:
    """Return the project root path."""
    return PROJECT_ROOT
