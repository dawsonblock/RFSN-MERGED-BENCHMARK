"""Incremental test running for faster patch evaluation.

This module provides:
1. Import tracing to find affected tests
2. Smart test selection based on changed files
3. Graduated test execution (affected → related → full)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# DIFF PARSING
# ============================================================================

def parse_changed_files(diff: str) -> list[str]:
    """Extract list of changed files from a unified diff.
    
    Args:
        diff: Unified diff string.
        
    Returns:
        List of file paths that were changed.
    """
    files = set()
    
    # Match diff headers: --- a/path or +++ b/path
    for line in diff.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line[6:].strip()
            if path != "/dev/null":
                files.add(path)
        elif line.startswith("diff --git"):
            # Alternative format: diff --git a/file b/file
            match = re.search(r"diff --git a/(.+?) b/(.+)", line)
            if match:
                files.add(match.group(2))
    
    return list(files)


def parse_changed_functions(diff: str) -> dict[str, list[str]]:
    """Extract changed functions/classes from a diff.
    
    Args:
        diff: Unified diff string.
        
    Returns:
        Dict mapping file paths to lists of changed function/class names.
    """
    changes: dict[str, list[str]] = {}
    current_file = None
    
    for line in diff.split("\n"):
        # Track current file
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            changes[current_file] = []
        
        # Look for function/class definitions in added lines
        if line.startswith("+") and not line.startswith("+++"):
            content = line[1:]
            
            # Python function
            match = re.match(r"\s*(async\s+)?def\s+(\w+)", content)
            if match and current_file:
                changes[current_file].append(match.group(2))
            
            # Python class
            match = re.match(r"\s*class\s+(\w+)", content)
            if match and current_file:
                changes[current_file].append(match.group(1))
    
    return changes


# ============================================================================
# IMPORT GRAPH
# ============================================================================

@dataclass
class ImportGraph:
    """Graph of Python imports for dependency tracking."""
    
    # file -> set of files it imports
    imports: dict[str, set[str]] = field(default_factory=dict)
    # file -> set of files that import it
    importers: dict[str, set[str]] = field(default_factory=dict)
    # Module name -> file path mapping
    modules: dict[str, str] = field(default_factory=dict)
    
    def add_import(self, source: str, target: str) -> None:
        """Record that source imports target."""
        if source not in self.imports:
            self.imports[source] = set()
        self.imports[source].add(target)
        
        if target not in self.importers:
            self.importers[target] = set()
        self.importers[target].add(source)
    
    def get_dependents(self, file: str, max_depth: int = 3) -> set[str]:
        """Get all files that depend on the given file.
        
        Args:
            file: The file to find dependents for.
            max_depth: Maximum dependency depth to traverse.
            
        Returns:
            Set of file paths that depend on the given file.
        """
        dependents = set()
        queue = [(file, 0)]
        visited = {file}
        
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for importer in self.importers.get(current, []):
                if importer not in visited:
                    visited.add(importer)
                    dependents.add(importer)
                    queue.append((importer, depth + 1))
        
        return dependents


def build_import_graph(repo_dir: str) -> ImportGraph:
    """Build an import graph for a Python repository.
    
    Args:
        repo_dir: Path to the repository root.
        
    Returns:
        ImportGraph with import relationships.
    """
    graph = ImportGraph()
    repo_path = Path(repo_dir)
    
    # Find all Python files
    py_files = list(repo_path.rglob("*.py"))
    
    # Build module name -> file path mapping
    for py_file in py_files:
        rel_path = py_file.relative_to(repo_path)
        
        # Convert path to module name
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py
        
        module_name = ".".join(parts)
        graph.modules[module_name] = str(rel_path)
    
    # Parse imports from each file
    for py_file in py_files:
        rel_path = str(py_file.relative_to(repo_path))
        
        try:
            with open(py_file, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = graph.modules.get(alias.name)
                        if target:
                            graph.add_import(rel_path, target)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        target = graph.modules.get(node.module)
                        if target:
                            graph.add_import(rel_path, target)
        
        except (SyntaxError, UnicodeDecodeError):
            continue
    
    return graph


# ============================================================================
# TEST DISCOVERY
# ============================================================================

def find_test_files(repo_dir: str) -> list[str]:
    """Find all test files in a repository.
    
    Args:
        repo_dir: Path to the repository root.
        
    Returns:
        List of test file paths (relative to repo_dir).
    """
    test_files = []
    repo_path = Path(repo_dir)
    
    # Common test patterns
    patterns = [
        "test_*.py",
        "*_test.py",
        "tests/**/*.py",
        "test/**/*.py",
    ]
    
    for pattern in patterns:
        for match in repo_path.glob(pattern):
            if match.is_file() and not match.name.startswith("__"):
                test_files.append(str(match.relative_to(repo_path)))
    
    # Also check for conftest.py files
    for conftest in repo_path.rglob("conftest.py"):
        test_files.append(str(conftest.relative_to(repo_path)))
    
    return list(set(test_files))


def find_tests_for_file(
    changed_file: str,
    test_files: list[str],
    graph: ImportGraph | None = None,
) -> list[str]:
    """Find test files that likely test a given source file.
    
    Args:
        changed_file: The source file that changed.
        test_files: List of all test files in the repo.
        graph: Optional import graph for dependency analysis.
        
    Returns:
        List of test files that likely test the changed file.
    """
    related_tests = []
    changed_path = Path(changed_file)
    changed_name = changed_path.stem
    
    for test_file in test_files:
        test_path = Path(test_file)
        test_name = test_path.stem
        
        # Pattern matching
        # test_foo.py tests foo.py
        if test_name == f"test_{changed_name}":
            related_tests.append(test_file)
            continue
        
        # foo_test.py tests foo.py
        if test_name == f"{changed_name}_test":
            related_tests.append(test_file)
            continue
        
        # Same directory heuristic
        if test_path.parent == changed_path.parent:
            if "test" in test_name.lower():
                related_tests.append(test_file)
                continue
    
    # Use import graph if available
    if graph:
        dependents = graph.get_dependents(changed_file)
        for dep in dependents:
            if dep in test_files and dep not in related_tests:
                related_tests.append(dep)
    
    return related_tests


# ============================================================================
# INCREMENTAL TEST SELECTION
# ============================================================================

@dataclass
class TestSelection:
    """Result of incremental test selection."""
    
    # Tests directly related to changed files
    affected_tests: list[str] = field(default_factory=list)
    
    # Tests in the same package/directory
    related_tests: list[str] = field(default_factory=list)
    
    # All other test files
    remaining_tests: list[str] = field(default_factory=list)
    
    # Changed source files
    changed_files: list[str] = field(default_factory=list)
    
    def get_focused_command(self, framework: str = "pytest") -> str:
        """Get command to run only affected tests.
        
        Args:
            framework: Test framework (pytest, jest, cargo, etc.)
            
        Returns:
            Command string.
        """
        if not self.affected_tests:
            return ""
        
        if framework == "pytest":
            files = " ".join(self.affected_tests[:5])  # Limit to 5 files
            return f"pytest -q {files}"
        
        elif framework == "jest":
            patterns = [f"--testPathPattern={t}" for t in self.affected_tests[:3]]
            return f"jest {' '.join(patterns)}"
        
        elif framework == "cargo":
            # Cargo test runs all by default, use --test for specific
            return "cargo test"
        
        else:
            return ""
    
    def get_staged_commands(self, framework: str = "pytest") -> list[tuple[str, str]]:
        """Get staged test commands (affected → related → full).
        
        Returns:
            List of (stage_name, command) tuples.
        """
        commands = []
        
        if framework == "pytest":
            if self.affected_tests:
                files = " ".join(self.affected_tests[:5])
                commands.append(("affected", f"pytest -q {files}"))
            
            if self.related_tests:
                files = " ".join(self.related_tests[:5])
                commands.append(("related", f"pytest -q {files}"))
            
            commands.append(("full", "pytest -q"))
        
        return commands


def select_tests_for_patch(
    diff: str,
    repo_dir: str,
    graph: ImportGraph | None = None,
) -> TestSelection:
    """Select tests to run for a given patch.
    
    Args:
        diff: The unified diff of the patch.
        repo_dir: Path to the repository root.
        graph: Optional pre-built import graph.
        
    Returns:
        TestSelection with prioritized test lists.
    """
    selection = TestSelection()
    
    # Parse changed files from diff
    selection.changed_files = parse_changed_files(diff)
    
    if not selection.changed_files:
        return selection
    
    # Find all test files
    all_tests = find_test_files(repo_dir)
    
    # Build import graph if not provided
    if graph is None:
        try:
            graph = build_import_graph(repo_dir)
        except Exception:
            graph = ImportGraph()
    
    # Find affected tests
    affected = set()
    for changed_file in selection.changed_files:
        tests = find_tests_for_file(changed_file, all_tests, graph)
        affected.update(tests)
    
    selection.affected_tests = list(affected)
    
    # Find related tests (same directory as changed files)
    changed_dirs = {str(Path(f).parent) for f in selection.changed_files}
    related = set()
    for test in all_tests:
        if test not in affected:
            test_dir = str(Path(test).parent)
            if test_dir in changed_dirs or any(
                test_dir.startswith(d) or d.startswith(test_dir)
                for d in changed_dirs
            ):
                related.add(test)
    
    selection.related_tests = list(related)
    
    # Remaining tests
    used = affected | related
    selection.remaining_tests = [t for t in all_tests if t not in used]
    
    return selection


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_incremental_test_command(
    diff: str,
    repo_dir: str,
    framework: str = "pytest",
    fallback_cmd: str = "pytest -q",
) -> str:
    """Get the optimal test command for a patch.
    
    If the patch affects specific files, returns a focused command.
    Otherwise returns the fallback command.
    
    Args:
        diff: The unified diff of the patch.
        repo_dir: Path to the repository root.
        framework: Test framework.
        fallback_cmd: Command to use if no focused tests found.
        
    Returns:
        Test command string.
    """
    try:
        selection = select_tests_for_patch(diff, repo_dir)
        focused = selection.get_focused_command(framework)
        return focused if focused else fallback_cmd
    except Exception:
        return fallback_cmd


def should_skip_full_tests(
    diff: str,
    repo_dir: str,
    affected_threshold: int = 3,
) -> bool:
    """Determine if we can skip full test suite.
    
    If fewer than `affected_threshold` test files are affected,
    we might be able to skip the full suite after focused tests pass.
    
    Args:
        diff: The unified diff.
        repo_dir: Path to the repository.
        affected_threshold: Max affected tests to consider "small change".
        
    Returns:
        True if full tests can potentially be skipped.
    """
    try:
        selection = select_tests_for_patch(diff, repo_dir)
        total_affected = len(selection.affected_tests) + len(selection.related_tests)
        return total_affected <= affected_threshold
    except Exception:
        return False
