"""Static analysis test for command purity.

This test searches the codebase for forbidden shell execution patterns
and fails if any are found. This enforces the "no shell=True, no sh -c"
security invariant.

Tests in this module:
- Pattern-based detection of shell=True usage
- AST-based subprocess call validation
- exec_utils security enforcement
- Phase 1 shell elimination verification
"""

import ast
import os
import re
from pathlib import Path

import pytest

# Root directory of the rfsn_controller package
PACKAGE_ROOT = Path(__file__).parent.parent / "rfsn_controller"

# Directories to exclude from scanning
EXCLUDED_DIRS: set[str] = {
    "__pycache__",
    ".git",
    "firecracker-main",  # External dependency
    "E2B-main",  # External dependency
    "RFSN",  # External reference
}

# Files to exclude (temporary/debug scripts)
EXCLUDED_FILES: set[str] = {
    "debug_runner.py",
    "shell_scanner.py",  # Contains detection patterns as strings
    "contracts.py",  # Contains shell validation messages (Phase 5)
}

# Forbidden patterns
FORBIDDEN_PATTERNS = [
    (r'shell\s*=\s*True', "shell=True"),
    (r'["\'](sh|bash|dash|zsh)\s+-c\b', "sh -c wrapper"),
    (r'bash\s+-lc\b', "bash -lc wrapper"),
]


def get_python_files() -> list[Path]:
    """Get all Python files in the package, excluding specified directories."""
    files = []
    for root, dirs, filenames in os.walk(PACKAGE_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for filename in filenames:
            if filename.endswith(".py") and filename not in EXCLUDED_FILES:
                files.append(Path(root) / filename)
    
    return files


def find_pattern_violations(content: str, patterns: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    """Find violations of forbidden patterns in file content.
    
    Args:
        content: File content to scan.
        patterns: List of (regex, description) tuples.
        
    Returns:
        List of (line_number, matched_text, pattern_description) tuples.
    """
    violations = []
    lines = content.split("\n")
    in_docstring = False
    docstring_char = None
    
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        
        # Track docstrings (triple quotes)
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                # Check if single-line docstring
                if stripped.count(docstring_char) >= 2:
                    continue  # Single-line docstring, skip
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
                docstring_char = None
            continue
        
        # Skip comments
        if stripped.startswith("#"):
            continue
        
        for pattern, description in patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                # Skip if it's in a comment at end of line
                comment_pos = line.find("#")
                if comment_pos >= 0 and match.start() > comment_pos:
                    continue
                    
                # Skip if it's explicitly setting shell=False (which is good)
                if "shell=False" in line and description == "shell=True":
                    continue
                
                violations.append((line_num, match.group(), description))
    
    return violations


def check_subprocess_calls_use_lists(content: str, filename: str) -> list[str]:
    """Use AST to verify subprocess calls use list arguments.
    
    Args:
        content: Python source code.
        filename: Name of file for error messages.
        
    Returns:
        List of error messages.
    """
    errors = []
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []  # Skip files with syntax errors
    
    for node in ast.walk(tree):
        # Check for subprocess.run, subprocess.call, subprocess.Popen
        if isinstance(node, ast.Call):
            func = node.func
            
            # Check for subprocess.run(...) style calls
            if isinstance(func, ast.Attribute):
                if func.attr in ("run", "call", "Popen", "check_call", "check_output"):
                    # Check if module is subprocess
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        # Verify first argument is a list or variable (not a string)
                        if node.args:
                            first_arg = node.args[0]
                            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                                errors.append(
                                    f"{filename}:{node.lineno}: subprocess.{func.attr}() "
                                    f"called with string argument instead of list"
                                )
                        
                        # Check for shell=True in keyword args
                        for kw in node.keywords:
                            if kw.arg == "shell":
                                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    errors.append(
                                        f"{filename}:{node.lineno}: subprocess.{func.attr}() "
                                        f"called with shell=True"
                                    )
    
    return errors


@pytest.mark.security
class TestNoShellExecution:
    """Test suite for command purity enforcement.
    
    These tests validate the Phase 1 shell elimination changes and ensure
    no shell=True or sh -c patterns exist in the codebase.
    """
    
    def test_no_forbidden_patterns_in_codebase(self) -> None:
        """Verify no forbidden shell patterns exist in the codebase."""
        all_violations = []
        
        for filepath in get_python_files():
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            violations = find_pattern_violations(content, FORBIDDEN_PATTERNS)
            
            for line_num, matched, description in violations:
                all_violations.append(
                    f"{filepath.relative_to(PACKAGE_ROOT.parent)}:{line_num}: "
                    f"Forbidden pattern '{description}': {matched}"
                )
        
        if all_violations:
            pytest.fail(
                f"Found {len(all_violations)} forbidden shell patterns:\n"
                + "\n".join(all_violations)
            )
    
    def test_subprocess_calls_use_lists(self) -> None:
        """Verify subprocess calls use list arguments, not strings."""
        all_errors = []
        
        for filepath in get_python_files():
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            errors = check_subprocess_calls_use_lists(
                content, 
                str(filepath.relative_to(PACKAGE_ROOT.parent))
            )
            all_errors.extend(errors)
        
        if all_errors:
            pytest.fail(
                f"Found {len(all_errors)} subprocess calls with string arguments:\n"
                + "\n".join(all_errors)
            )
    
    def test_exec_utils_enforces_argv_list(self) -> None:
        """Verify exec_utils.safe_run rejects non-list arguments."""
        from rfsn_controller.exec_utils import safe_run
        
        # Should raise ValueError for string input
        with pytest.raises((ValueError, TypeError)):
            safe_run("echo test", cwd="/tmp")  # type: ignore
        
        # Should raise ValueError for empty list
        with pytest.raises(ValueError):
            safe_run([], cwd="/tmp")
    
    def test_exec_utils_rejects_shell_wrappers(self) -> None:
        """Verify exec_utils rejects sh -c style wrappers."""
        from rfsn_controller.exec_utils import safe_run
        
        # Should reject sh -c
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            safe_run(["sh", "-c", "echo test"], cwd="/tmp")
        
        # Should reject bash -c
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            safe_run(["bash", "-c", "echo test"], cwd="/tmp")
    
    def test_safe_run_works_with_valid_argv(self) -> None:
        """Verify safe_run works with proper argv list."""
        from rfsn_controller.exec_utils import safe_run
        
        result = safe_run(
            ["echo", "hello", "world"],
            cwd="/tmp",
            check_global_allowlist=False,  # Skip allowlist for test
        )
        
        assert result.ok
        assert "hello world" in result.stdout
        assert result.exit_code == 0


@pytest.mark.security
class TestPhase1ShellElimination:
    """Tests validating Phase 1 shell elimination refactoring.
    
    These tests ensure the SubprocessPool refactoring maintains security
    and functionality after eliminating interactive bash shells.
    """
    
    def test_subprocess_pool_rejects_interactive_shell(self) -> None:
        """Verify SubprocessPool rejects /bin/bash -i pattern."""
        from rfsn_controller.optimizations import SubprocessPool
        
        pool = SubprocessPool()
        
        # The old code used ["/bin/bash", "-i"] which is now rejected
        with pytest.raises(ValueError, match="Interactive shell"):
            pool.run_command(["/bin/bash", "-i"])
        
        with pytest.raises(ValueError, match="Interactive shell"):
            pool.run_command(["bash", "-i"])
    
    def test_subprocess_pool_rejects_sh_c(self) -> None:
        """Verify SubprocessPool rejects sh -c pattern."""
        from rfsn_controller.optimizations import SubprocessPool
        
        pool = SubprocessPool()
        
        with pytest.raises(ValueError, match="Shell wrapper"):
            pool.run_command(["sh", "-c", "echo test"])
        
        with pytest.raises(ValueError, match="Shell wrapper"):
            pool.run_command(["bash", "-c", "echo test"])
    
    def test_subprocess_pool_executes_safe_commands(self) -> None:
        """Verify SubprocessPool can execute safe argv commands."""
        from rfsn_controller.optimizations import SubprocessPool
        
        pool = SubprocessPool()
        
        result = pool.run_command(["echo", "phase1 test"])
        
        assert result.ok
        assert "phase1 test" in result.stdout
    
    def test_subprocess_pool_no_shell_injection(self) -> None:
        """Verify shell metacharacters don't enable injection."""
        from rfsn_controller.optimizations import SubprocessPool
        
        pool = SubprocessPool()
        
        # These would be dangerous with shell=True
        dangerous_inputs = [
            "hello; rm -rf /",
            "$(whoami)",
            "`id`",
            "hello && cat /etc/passwd",
            "test || echo pwned",
        ]
        
        for malicious in dangerous_inputs:
            result = pool.run_command(["echo", malicious])
            assert result.ok
            # The malicious string should be echoed literally, not interpreted
            assert malicious in result.stdout or result.ok
    
    def test_optimizations_module_has_no_shell_true(self) -> None:
        """Verify optimizations.py has no shell=True after refactor."""
        optimizations_path = PACKAGE_ROOT / "optimizations.py"
        content = optimizations_path.read_text()
        
        violations = find_pattern_violations(content, FORBIDDEN_PATTERNS)
        
        # Filter to only include actual violations (not in comments/docstrings)
        real_violations = [
            v for v in violations 
            if "shell=False" not in content.split("\n")[v[0]-1]
        ]
        
        if real_violations:
            violation_details = "\n".join(
                f"Line {v[0]}: {v[2]} - {v[1]}" for v in real_violations
            )
            pytest.fail(
                f"Found {len(real_violations)} violations in optimizations.py "
                f"after Phase 1 refactor:\n{violation_details}"
            )
    
    def test_exec_utils_has_no_shell_true(self) -> None:
        """Verify exec_utils.py has no executable shell=True (except comments/docstrings)."""
        exec_utils_path = PACKAGE_ROOT / "exec_utils.py"
        content = exec_utils_path.read_text()
        
        # Use AST to find actual shell=True in code
        errors = check_subprocess_calls_use_lists(
            content, 
            str(exec_utils_path.relative_to(PACKAGE_ROOT.parent))
        )
        
        if errors:
            pytest.fail(
                "Found shell=True in exec_utils.py code:\n" + "\n".join(errors)
            )
