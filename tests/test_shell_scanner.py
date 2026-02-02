"""Tests for the shell_scanner utility.

This test module verifies that shell_scanner correctly identifies:
- shell=True usage in subprocess calls
- Interactive shell patterns
- Shell wrapper patterns (sh -c, bash -c)
- os.system and os.popen calls

Tests use the @pytest.mark.scanner marker for selective execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfsn_controller.shell_scanner import (
    ScanResult,
    ShellScanner,
    Violation,
    discover_python_files,
    format_github_actions,
    format_json,
    format_text,
    main,
    scan_with_ast,
    scan_with_regex,
)

# =============================================================================
# Scanner Core Tests
# =============================================================================

@pytest.mark.scanner
class TestShellScanner:
    """Test the ShellScanner class."""
    
    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Scanner handles empty directories gracefully."""
        scanner = ShellScanner()
        result = scanner.scan([tmp_path])
        
        assert result.files_scanned == 0
        assert not result.has_violations
        assert len(result.violations) == 0
    
    def test_scan_safe_code(
        self, 
        tmp_py_file, 
        safe_code_sample: str
    ) -> None:
        """Scanner accepts safe code without violations."""
        filepath = tmp_py_file("safe.py", safe_code_sample)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        assert result.files_scanned == 1
        assert not result.has_violations
    
    def test_scan_detects_shell_true(self, tmp_py_file) -> None:
        """Scanner detects shell=True violations."""
        code = '''
import subprocess

def bad():
    subprocess.run("echo hello", shell=True)
'''
        filepath = tmp_py_file("bad.py", code)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        assert result.has_violations
        assert result.critical_count >= 1
        assert any("shell" in v.category.lower() for v in result.violations)
    
    def test_scan_detects_os_system(self, tmp_py_file) -> None:
        """Scanner detects os.system calls."""
        code = '''
import os

def danger():
    os.system("rm -rf /")
'''
        filepath = tmp_py_file("danger.py", code)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        assert result.has_violations
        assert any(v.category == "os_system" for v in result.violations)
    
    def test_scan_detects_shell_wrapper(self, tmp_py_file) -> None:
        """Scanner detects sh -c / bash -c wrappers."""
        code = '''
import subprocess

def wrapper():
    subprocess.run(["sh", "-c", "echo test"])
    subprocess.run(["bash", "-c", "rm stuff"])
'''
        filepath = tmp_py_file("wrapper.py", code)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        assert result.has_violations
        assert any("wrapper" in v.category.lower() for v in result.violations)
    
    def test_scan_detects_interactive_shell(self, tmp_py_file) -> None:
        """Scanner detects interactive shell invocations."""
        code = '''
import subprocess

def interactive():
    subprocess.Popen(["/bin/bash", "-i"])
'''
        filepath = tmp_py_file("interactive.py", code)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        assert result.has_violations
    
    def test_scan_ignores_comments(self, tmp_py_file) -> None:
        """Scanner ignores violations in comments."""
        code = '''
import subprocess

# Don't use shell=True - it's dangerous
# subprocess.run("cmd", shell=True) # This is bad

def safe():
    subprocess.run(["echo", "hi"], shell=False)
'''
        filepath = tmp_py_file("commented.py", code)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        assert not result.has_violations
    
    def test_scan_ignores_docstrings(self, tmp_py_file) -> None:
        """Scanner ignores violations in docstrings."""
        code = '''
import subprocess

def documented():
    """
    Example of what NOT to do:
    subprocess.run("cmd", shell=True)
    """
    return subprocess.run(["echo", "safe"], shell=False)
'''
        filepath = tmp_py_file("docstring.py", code)
        
        scanner = ShellScanner()
        result = scanner.scan([filepath])
        
        # Should not flag the example in docstring
        assert not result.has_violations
    
    def test_scan_multiple_files(
        self, 
        mixed_code_project: Path
    ) -> None:
        """Scanner correctly processes multiple files."""
        scanner = ShellScanner()
        result = scanner.scan([mixed_code_project])
        
        assert result.files_scanned >= 3
        assert result.has_violations
    
    def test_scan_exclude_directories(
        self, 
        tmp_project_dir
    ) -> None:
        """Scanner respects directory exclusions."""
        project = tmp_project_dir({
            "main.py": "import subprocess; subprocess.run('ls', shell=True)",
            "vendor/lib.py": "import subprocess; subprocess.run('ls', shell=True)",
        })
        
        scanner = ShellScanner(exclude_dirs={"vendor"})
        result = scanner.scan([project])
        
        # Should only scan main.py, not vendor/lib.py
        assert result.files_scanned == 1
    
    def test_scan_exclude_files(self, tmp_project_dir) -> None:
        """Scanner respects file exclusions."""
        project = tmp_project_dir({
            "main.py": "import subprocess; subprocess.run('ls', shell=True)",
            "test_main.py": "import subprocess; subprocess.run('ls', shell=True)",
        })
        
        scanner = ShellScanner(exclude_files={"test_main.py"})
        result = scanner.scan([project])
        
        assert result.files_scanned == 1
    
    def test_scan_exclude_patterns(self, tmp_project_dir) -> None:
        """Scanner respects regex exclusion patterns."""
        project = tmp_project_dir({
            "main.py": "import subprocess; subprocess.run('ls', shell=True)",
            "generated_code.py": "import subprocess; subprocess.run('ls', shell=True)",
        })
        
        scanner = ShellScanner(exclude_patterns=["generated_.*"])
        result = scanner.scan([project])
        
        assert result.files_scanned == 1


# =============================================================================
# AST Scanner Tests
# =============================================================================

@pytest.mark.scanner
class TestASTScanner:
    """Test AST-based detection."""
    
    def test_ast_detects_subprocess_shell_true(self) -> None:
        """AST scanner detects subprocess with shell=True."""
        code = '''
import subprocess
subprocess.run(["ls"], shell=True)
'''
        violations = scan_with_ast(Path("test.py"), code)
        
        assert len(violations) >= 1
        assert any("shell" in v.category.lower() for v in violations)
    
    def test_ast_detects_string_argument(self) -> None:
        """AST scanner flags string arguments without shell kwarg."""
        code = '''
import subprocess
subprocess.run("ls -la")
'''
        violations = scan_with_ast(Path("test.py"), code)
        
        assert any("string" in v.category.lower() for v in violations)
    
    def test_ast_detects_shell_wrapper_in_list(self) -> None:
        """AST scanner detects shell wrappers in argv."""
        code = '''
import subprocess
subprocess.run(["sh", "-c", "echo test"])
'''
        violations = scan_with_ast(Path("test.py"), code)
        
        assert any("wrapper" in v.category.lower() for v in violations)
    
    def test_ast_handles_syntax_errors(self) -> None:
        """AST scanner gracefully handles syntax errors."""
        code = "def broken(:"  # Invalid syntax
        violations = scan_with_ast(Path("broken.py"), code)
        
        # Should return empty list, not raise
        assert violations == []


# =============================================================================
# Regex Scanner Tests
# =============================================================================

@pytest.mark.scanner
class TestRegexScanner:
    """Test regex-based detection."""
    
    def test_regex_detects_shell_true(self) -> None:
        """Regex scanner detects shell=True pattern."""
        code = 'subprocess.run(cmd, shell=True)'
        violations = scan_with_regex(Path("test.py"), code)
        
        assert any(v.category == "shell_true" for v in violations)
    
    def test_regex_detects_os_popen(self) -> None:
        """Regex scanner detects os.popen."""
        code = 'os.popen("ls -la")'
        violations = scan_with_regex(Path("test.py"), code)
        
        assert any(v.category == "os_popen" for v in violations)
    
    def test_regex_detects_interactive_bash(self) -> None:
        """Regex scanner detects interactive bash."""
        code = 'subprocess.Popen(["/bin/bash", "-i"])'
        violations = scan_with_regex(Path("test.py"), code)
        
        assert any("interactive" in v.category.lower() for v in violations)
    
    def test_regex_respects_shell_false(self) -> None:
        """Regex scanner doesn't flag shell=False."""
        code = 'subprocess.run(cmd, shell=False)'
        violations = scan_with_regex(Path("test.py"), code)
        
        # shell=False should not be flagged
        assert not any(v.category == "shell_true" for v in violations)


# =============================================================================
# File Discovery Tests
# =============================================================================

@pytest.mark.scanner
class TestFileDiscovery:
    """Test file discovery functionality."""
    
    def test_discovers_python_files(self, tmp_project_dir) -> None:
        """Discovers .py files in directories."""
        project = tmp_project_dir({
            "main.py": "",
            "utils.py": "",
            "sub/module.py": "",
        })
        
        files = list(discover_python_files([project]))
        
        assert len(files) == 3
        assert all(f.suffix == ".py" for f in files)
    
    def test_excludes_pycache(self, tmp_project_dir) -> None:
        """Excludes __pycache__ directories."""
        project = tmp_project_dir({
            "main.py": "",
            "__pycache__/cached.py": "",
        })
        
        files = list(discover_python_files([project]))
        
        assert len(files) == 1
        assert "__pycache__" not in str(files[0])
    
    def test_handles_nonexistent_paths(self) -> None:
        """Handles nonexistent paths gracefully."""
        files = list(discover_python_files([Path("/nonexistent/path")]))
        assert files == []


# =============================================================================
# Output Format Tests
# =============================================================================

@pytest.mark.scanner
class TestOutputFormats:
    """Test output formatting functions."""
    
    @pytest.fixture
    def sample_result(self) -> ScanResult:
        """Create a sample result for testing formatters."""
        return ScanResult(
            files_scanned=5,
            violations=[
                Violation(
                    file="test.py",
                    line=10,
                    column=0,
                    category="shell_true",
                    severity="critical",
                    message="shell=True detected",
                    snippet="subprocess.run(cmd, shell=True)",
                ),
                Violation(
                    file="test.py",
                    line=20,
                    column=4,
                    category="os_system",
                    severity="critical",
                    message="os.system detected",
                    snippet="os.system('ls')",
                ),
            ],
        )
    
    def test_format_text(self, sample_result: ScanResult) -> None:
        """Text formatter produces readable output."""
        output = format_text(sample_result)
        
        assert "Files scanned: 5" in output
        assert "Violations found: 2" in output
        assert "Critical: 2" in output
        assert "test.py:10" in output
    
    def test_format_text_verbose(self, sample_result: ScanResult) -> None:
        """Text formatter shows snippets in verbose mode."""
        output = format_text(sample_result, verbose=True)
        
        assert "subprocess.run" in output or "shell=True" in output
    
    def test_format_json(self, sample_result: ScanResult) -> None:
        """JSON formatter produces valid JSON."""
        output = format_json(sample_result)
        data = json.loads(output)
        
        assert data["files_scanned"] == 5
        assert data["violation_count"] == 2
        assert len(data["violations"]) == 2
    
    def test_format_github_actions(self, sample_result: ScanResult) -> None:
        """GitHub Actions formatter produces workflow commands."""
        output = format_github_actions(sample_result)
        
        assert "::error file=test.py" in output
        assert "line=10" in output


# =============================================================================
# CLI Tests
# =============================================================================

@pytest.mark.scanner
class TestCLI:
    """Test command-line interface."""
    
    def test_cli_scan_directory(self, tmp_py_file, monkeypatch) -> None:
        """CLI can scan a directory."""
        filepath = tmp_py_file("clean.py", "print('hello')")
        
        exit_code = main([str(filepath.parent)])
        
        assert exit_code == 0
    
    def test_cli_ci_mode_violations(self, tmp_py_file) -> None:
        """CLI returns exit code 1 in CI mode with violations."""
        filepath = tmp_py_file(
            "bad.py", 
            "import subprocess; subprocess.run('x', shell=True)"
        )
        
        exit_code = main(["--ci", str(filepath)])
        
        assert exit_code == 1
    
    def test_cli_ci_mode_clean(self, tmp_py_file) -> None:
        """CLI returns exit code 0 in CI mode without violations."""
        filepath = tmp_py_file(
            "good.py", 
            "import subprocess; subprocess.run(['x'], shell=False)"
        )
        
        exit_code = main(["--ci", str(filepath)])
        
        assert exit_code == 0
    
    def test_cli_json_output(self, tmp_py_file, capsys) -> None:
        """CLI can produce JSON output."""
        filepath = tmp_py_file("test.py", "print('hi')")
        
        main(["--json", str(filepath)])
        
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "files_scanned" in data
    
    def test_cli_nonexistent_path(self, capsys) -> None:
        """CLI handles nonexistent paths."""
        exit_code = main(["/nonexistent/path/to/code"])
        
        assert exit_code == 1


# =============================================================================
# ScanResult Tests
# =============================================================================

@pytest.mark.scanner
class TestScanResult:
    """Test ScanResult dataclass."""
    
    def test_has_violations(self) -> None:
        """has_violations property works correctly."""
        empty = ScanResult()
        assert not empty.has_violations
        
        with_violations = ScanResult(violations=[
            Violation("f", 1, 0, "cat", "high", "msg")
        ])
        assert with_violations.has_violations
    
    def test_severity_counts(self) -> None:
        """Severity count properties work correctly."""
        result = ScanResult(violations=[
            Violation("f", 1, 0, "c1", "critical", "msg"),
            Violation("f", 2, 0, "c2", "critical", "msg"),
            Violation("f", 3, 0, "c3", "high", "msg"),
            Violation("f", 4, 0, "c4", "medium", "msg"),
        ])
        
        assert result.critical_count == 2
        assert result.high_count == 1
    
    def test_to_dict(self) -> None:
        """to_dict produces correct dictionary."""
        result = ScanResult(
            files_scanned=10,
            violations=[Violation("f", 1, 0, "c", "high", "msg")],
            errors=["error1"],
        )
        
        d = result.to_dict()
        
        assert d["files_scanned"] == 10
        assert d["violation_count"] == 1
        assert len(d["errors"]) == 1


# =============================================================================
# Integration with Codebase
# =============================================================================

@pytest.mark.scanner
@pytest.mark.security
class TestCodebaseIntegration:
    """Test scanner against actual rfsn_controller codebase."""
    
    def test_scan_rfsn_controller_no_violations(
        self, 
        rfsn_controller_path: Path
    ) -> None:
        """The rfsn_controller package should have no shell violations.
        
        This validates that Phase 1 shell elimination was successful.
        
        Known Exceptions:
        - shell_scanner.py: Contains pattern strings for detection
        - sandbox.py: Uses sh -c for Docker container execution (safe context)
        """
        scanner = ShellScanner(
            exclude_dirs={"__pycache__", "buildpacks"},
            exclude_files={
                "shell_scanner.py",  # Contains detection patterns as strings
            },
        )
        result = scanner.scan([rfsn_controller_path])
        
        # Filter out known safe patterns (Docker container shell execution)
        # The sandbox.py uses sh -c to run commands inside Docker containers,
        # which is a different security context than host shell execution
        safe_file_patterns = {
            "sandbox.py": {"shell_wrapper"},  # Docker container execution
            "contracts.py": {"shell_true"},  # Contains shell=True in error messages, not actual usage
        }
        
        critical_violations = [
            v for v in result.violations 
            if v.severity == "critical"
            and not (
                Path(v.file).name in safe_file_patterns
                and v.category in safe_file_patterns[Path(v.file).name]
            )
        ]
        
        if critical_violations:
            violation_details = "\n".join(str(v) for v in critical_violations)
            pytest.fail(
                f"Found {len(critical_violations)} critical violations in "
                f"rfsn_controller:\n{violation_details}"
            )
