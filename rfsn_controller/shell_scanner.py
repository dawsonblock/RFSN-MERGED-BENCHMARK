#!/usr/bin/env python3
"""Shell Scanner - Detect unsafe shell execution patterns in Python code.

This standalone utility scans Python files for:
- shell=True usage in subprocess calls
- Interactive shell patterns (/bin/bash -i, sh -c, etc.)
- Shell wrapper patterns that bypass security controls

Usage:
    # Scan a directory
    python -m rfsn_controller.shell_scanner /path/to/code
    
    # Scan specific files
    python -m rfsn_controller.shell_scanner file1.py file2.py
    
    # CI/CD mode (exit code 1 on violations)
    python -m rfsn_controller.shell_scanner --ci /path/to/code
    
    # JSON output for tooling integration
    python -m rfsn_controller.shell_scanner --json /path/to/code
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

__version__ = "1.0.0"


# =============================================================================
# Violation Types and Patterns
# =============================================================================

@dataclass
class Violation:
    """Represents a security violation found in source code."""
    
    file: str
    line: int
    column: int
    category: str
    severity: str  # "critical", "high", "medium", "low"
    message: str
    snippet: str = ""
    
    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: [{self.severity.upper()}] {self.message}"


@dataclass
class ScanResult:
    """Result of scanning a codebase."""
    
    files_scanned: int = 0
    violations: list[Violation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "critical")
    
    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "high")
    
    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "violation_count": len(self.violations),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "violations": [asdict(v) for v in self.violations],
            "errors": self.errors,
        }


# Regex patterns for shell usage detection
SHELL_PATTERNS: list[tuple[str, str, str, str]] = [
    # (pattern, category, severity, message)
    (r'\bshell\s*=\s*True\b', "shell_true", "critical",
     "shell=True enables shell injection vulnerabilities"),
    
    (r'["\'](/bin/)?(ba)?sh["\']\s*,\s*["\'](-c|-i)\b', "shell_wrapper", "critical",
     "Direct shell invocation with -c/-i flag detected"),
    
    (r'\[["\'](/bin/)?(ba)?sh["\'],\s*["\'](-[ci])', "shell_wrapper", "critical",
     "Shell wrapper in argv list detected"),
    
    (r'\bos\.system\s*\(', "os_system", "critical",
     "os.system() is inherently unsafe, use subprocess with shell=False"),
    
    (r'\bos\.popen\s*\(', "os_popen", "high",
     "os.popen() uses shell internally, use subprocess with shell=False"),
    
    (r'\bcommands\.(getoutput|getstatusoutput)\s*\(', "commands_module", "critical",
     "commands module is deprecated and unsafe"),
    
    (r'subprocess\.\w+\([^)]*shell\s*=\s*True', "subprocess_shell", "critical",
     "subprocess called with shell=True"),
    
    (r'Popen\([^)]*shell\s*=\s*True', "popen_shell", "critical",
     "Popen called with shell=True"),
    
    (r'/bin/bash[\"\']?,\s*[\"\']?-i', "interactive_bash", "critical",
     "Interactive bash shell detected - security risk"),
    
    (r'/bin/sh[\"\']?,\s*[\"\']?-i', "interactive_sh", "critical",
     "Interactive shell detected - security risk"),
    
    (r'(ba)?sh[\"\'],\s*[\"\']+-i', "interactive_shell", "critical",
     "Interactive shell pattern detected"),
    
    (r'\beval\s*\(\s*[^)]*(input|argv|environ)', "eval_user_input", "high",
     "eval() with user-controlled input detected"),
    
    (r'\bexec\s*\(\s*[^)]*(input|argv|environ)', "exec_user_input", "high",
     "exec() with user-controlled input detected"),
]

# Patterns that indicate safe usage (to avoid false positives)
SAFE_PATTERNS: list[str] = [
    r'shell\s*=\s*False',  # Explicit shell=False
    r'#.*shell\s*=\s*True',  # In comments
    r'["\']{3}.*shell\s*=\s*True.*["\']{3}',  # In docstrings
]


# =============================================================================
# AST-based Detection
# =============================================================================

class ShellUsageVisitor(ast.NodeVisitor):
    """AST visitor to detect shell usage patterns."""
    
    def __init__(self, filename: str, source_lines: list[str]):
        self.filename = filename
        self.source_lines = source_lines
        self.violations: list[Violation] = []
        
    def _get_line_snippet(self, lineno: int) -> str:
        """Get the source line as a snippet."""
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""
    
    def _add_violation(
        self, 
        node: ast.AST, 
        category: str, 
        severity: str, 
        message: str
    ) -> None:
        """Add a violation from an AST node."""
        lineno = getattr(node, 'lineno', 0)
        col_offset = getattr(node, 'col_offset', 0)
        self.violations.append(Violation(
            file=self.filename,
            line=lineno,
            column=col_offset,
            category=category,
            severity=severity,
            message=message,
            snippet=self._get_line_snippet(lineno),
        ))
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for shell usage."""
        # Check for subprocess.run/call/Popen/etc with shell=True
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("run", "call", "Popen", "check_call", "check_output"):
                # Check for shell=True keyword
                for kw in node.keywords:
                    if kw.arg == "shell":
                        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            self._add_violation(
                                node, "subprocess_shell_true", "critical",
                                f"subprocess.{node.func.attr}() called with shell=True"
                            )
                
                # Check for string argument (potential shell command)
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        # Check if shell=True or shell is not specified but using string
                        has_shell_kwarg = any(kw.arg == "shell" for kw in node.keywords)
                        if not has_shell_kwarg:
                            self._add_violation(
                                node, "string_command", "medium",
                                f"subprocess.{node.func.attr}() with string argument - "
                                "prefer argv list with explicit shell=False"
                            )
        
        # Check for os.system
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                if node.func.attr == "system":
                    self._add_violation(
                        node, "os_system", "critical",
                        "os.system() is inherently unsafe"
                    )
                elif node.func.attr == "popen":
                    self._add_violation(
                        node, "os_popen", "high",
                        "os.popen() uses shell internally"
                    )
        
        # Check for shell wrappers in argv
        if node.args:
            self._check_shell_wrapper_in_argv(node)
        
        self.generic_visit(node)
    
    def _check_shell_wrapper_in_argv(self, node: ast.Call) -> None:
        """Check if first argument is a shell wrapper pattern."""
        first_arg = node.args[0]
        
        # Check list literals like ["sh", "-c", ...]
        if isinstance(first_arg, ast.List):
            if len(first_arg.elts) >= 2:
                cmd = self._get_constant_value(first_arg.elts[0])
                flag = self._get_constant_value(first_arg.elts[1])
                
                if cmd and flag:
                    base_cmd = os.path.basename(cmd)
                    if base_cmd in ("sh", "bash", "dash", "zsh", "ksh"):
                        if flag in ("-c", "-i", "-ci", "-ic"):
                            self._add_violation(
                                node, "shell_wrapper_argv", "critical",
                                f"Shell wrapper [{base_cmd}, {flag}] detected in argv"
                            )
    
    def _get_constant_value(self, node: ast.AST) -> str | None:
        """Extract constant string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None


def scan_with_ast(filepath: Path, content: str) -> list[Violation]:
    """Scan a file using AST analysis.
    
    Args:
        filepath: Path to the file.
        content: File content.
        
    Returns:
        List of violations found.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []  # Skip files with syntax errors
    
    source_lines = content.split('\n')
    visitor = ShellUsageVisitor(str(filepath), source_lines)
    visitor.visit(tree)
    return visitor.violations


# =============================================================================
# Regex-based Detection
# =============================================================================

def scan_with_regex(filepath: Path, content: str) -> list[Violation]:
    """Scan a file using regex patterns.
    
    Args:
        filepath: Path to the file.
        content: File content.
        
    Returns:
        List of violations found.
    """
    violations: list[Violation] = []
    lines = content.split('\n')
    
    # Track if we're in a multi-line string (rough heuristic)
    in_docstring = False
    docstring_delimiter = None
    
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        
        # Skip empty lines and pure comments
        if not stripped or stripped.startswith('#'):
            continue
        
        # Track docstrings (simple heuristic)
        if not in_docstring:
            for delim in ('"""', "'''"):
                if delim in stripped:
                    count = stripped.count(delim)
                    if count == 1:
                        in_docstring = True
                        docstring_delimiter = delim
                    # If count >= 2, single-line docstring, check it but don't set flag
        else:
            if docstring_delimiter and docstring_delimiter in stripped:
                in_docstring = False
                docstring_delimiter = None
            continue  # Skip content in docstrings
        
        # Check each pattern
        for pattern, category, severity, message in SHELL_PATTERNS:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                # Check if match is in a comment
                comment_pos = line.find('#')
                if comment_pos >= 0 and match.start() > comment_pos:
                    continue
                
                # Check against safe patterns
                if any(re.search(sp, line, re.IGNORECASE) for sp in SAFE_PATTERNS):
                    continue
                
                violations.append(Violation(
                    file=str(filepath),
                    line=line_num,
                    column=match.start(),
                    category=category,
                    severity=severity,
                    message=message,
                    snippet=stripped[:100],
                ))
    
    return violations


# =============================================================================
# File Discovery
# =============================================================================

# Default directories to exclude
DEFAULT_EXCLUDES: set[str] = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    ".tox",
    ".nox",
    ".eggs",
    "*.egg-info",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
}


def discover_python_files(
    paths: list[Path],
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> Iterator[Path]:
    """Discover Python files from given paths.
    
    Args:
        paths: List of files or directories to scan.
        exclude_dirs: Directory names to exclude.
        exclude_files: File names to exclude.
        
    Yields:
        Path objects for Python files.
    """
    exclude_dirs = exclude_dirs or DEFAULT_EXCLUDES
    exclude_files = exclude_files or set()
    
    for path in paths:
        if path.is_file():
            if path.suffix == '.py' and path.name not in exclude_files:
                yield path
        elif path.is_dir():
            for root, dirs, files in os.walk(path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for filename in files:
                    if filename.endswith('.py') and filename not in exclude_files:
                        yield Path(root) / filename


# =============================================================================
# Main Scanner
# =============================================================================

class ShellScanner:
    """Main scanner class for detecting shell usage violations."""
    
    def __init__(
        self,
        exclude_dirs: set[str] | None = None,
        exclude_files: set[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        """Initialize the scanner.
        
        Args:
            exclude_dirs: Directory names to exclude.
            exclude_files: File names to exclude.
            exclude_patterns: Regex patterns for paths to exclude.
        """
        self.exclude_dirs = exclude_dirs or DEFAULT_EXCLUDES
        self.exclude_files = exclude_files or set()
        self.exclude_patterns = [re.compile(p) for p in (exclude_patterns or [])]
    
    def scan_file(self, filepath: Path) -> list[Violation]:
        """Scan a single file for violations.
        
        Args:
            filepath: Path to the Python file.
            
        Returns:
            List of violations found.
        """
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            return []  # Skip files we can't read
        
        # Combine AST and regex results, deduplicating
        ast_violations = scan_with_ast(filepath, content)
        regex_violations = scan_with_regex(filepath, content)
        
        # Deduplicate by (file, line, category)
        seen: set[tuple[str, int, str]] = set()
        violations: list[Violation] = []
        
        for v in ast_violations + regex_violations:
            key = (v.file, v.line, v.category)
            if key not in seen:
                seen.add(key)
                violations.append(v)
                
                # Log security violation event
                try:
                    from .events import log_security_violation_global
                    log_security_violation_global(
                        violation_type=v.category,
                        file_path=v.file,
                        line_number=v.line,
                        message=v.message,
                        severity=v.severity,
                    )
                except ImportError:
                    pass  # Events module not available
        
        return violations
    
    def scan(self, paths: list[Path]) -> ScanResult:
        """Scan paths for shell usage violations.
        
        Args:
            paths: List of files or directories to scan.
            
        Returns:
            ScanResult with all findings.
        """
        result = ScanResult()
        
        for filepath in discover_python_files(
            paths, 
            self.exclude_dirs, 
            self.exclude_files
        ):
            # Check exclude patterns
            if any(p.search(str(filepath)) for p in self.exclude_patterns):
                continue
            
            result.files_scanned += 1
            violations = self.scan_file(filepath)
            result.violations.extend(violations)
        
        return result


# =============================================================================
# Output Formatters
# =============================================================================

def format_text(result: ScanResult, verbose: bool = False) -> str:
    """Format scan result as human-readable text."""
    lines: list[str] = []
    
    lines.append(f"\n{'='*60}")
    lines.append("Shell Scanner Report")
    lines.append(f"{'='*60}\n")
    
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Violations found: {len(result.violations)}")
    lines.append(f"  Critical: {result.critical_count}")
    lines.append(f"  High: {result.high_count}")
    lines.append("")
    
    if result.violations:
        # Group by severity
        by_severity: dict[str, list[Violation]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }
        for v in result.violations:
            by_severity.setdefault(v.severity, []).append(v)
        
        for severity in ["critical", "high", "medium", "low"]:
            violations = by_severity.get(severity, [])
            if violations:
                lines.append(f"\n{severity.upper()} ({len(violations)} issues):")
                lines.append("-" * 40)
                for v in violations:
                    lines.append(f"  {v}")
                    if verbose and v.snippet:
                        lines.append(f"    > {v.snippet}")
    else:
        lines.append("✓ No shell usage violations found!")
    
    lines.append(f"\n{'='*60}\n")
    return "\n".join(lines)


def format_json(result: ScanResult) -> str:
    """Format scan result as JSON."""
    return json.dumps(result.to_dict(), indent=2)


def format_github_actions(result: ScanResult) -> str:
    """Format scan result for GitHub Actions annotations."""
    lines: list[str] = []
    
    for v in result.violations:
        # GitHub Actions workflow command format
        level = "error" if v.severity in ("critical", "high") else "warning"
        lines.append(
            f"::{level} file={v.file},line={v.line},col={v.column}::"
            f"[{v.category}] {v.message}"
        )
    
    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="shell_scanner",
        description="Scan Python code for unsafe shell execution patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/project           # Scan a directory
  %(prog)s file1.py file2.py          # Scan specific files
  %(prog)s --ci /path/to/code         # CI mode (exit 1 on violations)
  %(prog)s --json /path/to/code       # JSON output
  %(prog)s --format=github-actions .  # GitHub Actions format
        """,
    )
    
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to scan (default: current directory)",
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with code 1 if any violations found",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "github-actions"],
        default="text",
        help="Output format (default: text)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show code snippets for violations",
    )
    
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory name to exclude (can be used multiple times)",
    )
    
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        metavar="FILE",
        help="File name to exclude (can be used multiple times)",
    )
    
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="Regex pattern for paths to exclude",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).
        
    Returns:
        Exit code (0 = success, 1 = violations found or error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Build path list
    paths = [Path(p) for p in args.paths]
    
    # Validate paths exist
    for path in paths:
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            return 1
    
    # Build exclude sets
    exclude_dirs = DEFAULT_EXCLUDES | set(args.exclude_dir)
    exclude_files = set(args.exclude_file)
    
    # Create scanner and run
    scanner = ShellScanner(
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
        exclude_patterns=args.exclude_pattern,
    )
    
    result = scanner.scan(paths)
    
    # Format output
    if args.json or args.format == "json":
        output = format_json(result)
    elif args.format == "github-actions":
        output = format_github_actions(result)
    else:
        output = format_text(result, verbose=args.verbose)
    
    print(output)
    
    # Determine exit code
    if args.ci and result.has_violations:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
