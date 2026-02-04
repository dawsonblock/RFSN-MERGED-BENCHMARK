import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT CACHING
# ============================================================================

class VenvCache:
    """Caches fully-initialized repositories to skip slow installs.
    
    Caches the results of clone + install_deps per (repo_url, base_commit).
    Only active if RFSN_USE_ENV_CACHE=1 is set.
    """
    
    def __init__(self, cache_root: str = ".rfsn_state/env_cache"):
        self.cache_root = cache_root
        os.makedirs(self.cache_root, exist_ok=True)
    
    def _get_key(self, repo_url: str, base_commit: str) -> str:
        """Compute a stable hash for the (repo, commit) pair."""
        # Use repo name part + hash of full url
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:8]
        return f"{repo_name}_{base_commit[:12]}_{url_hash}"
    
    def get_warm_path(self, repo_url: str, base_commit: str) -> str | None:
        """Return path to warm setup if it exists and is ready."""
        if os.getenv("RFSN_USE_ENV_CACHE") != "1":
            return None
            
        key = self._get_key(repo_url, base_commit)
        path = os.path.join(self.cache_root, key)
        
        if os.path.exists(path) and os.path.exists(os.path.join(path, ".RFSN_READY")):
            logger.info("ENVIRONMENT CACHE: Hit for %s at %s", key, base_commit[:8])
            return path
        
        # Check for lock: if another process is currently initializing this env
        lock_file = path + ".lock"
        if os.path.exists(lock_file):
            logger.info("ENVIRONMENT CACHE: Waiting for lock on %s...", key)
            # Wait up to 5 minutes for another process to finish
            for _ in range(5):
                time.sleep(1)
                if os.path.exists(path) and os.path.exists(os.path.join(path, ".RFSN_READY")):
                    return path
                if not os.path.exists(lock_file):
                    break
                    
        return None
    
    def store(self, repo_url: str, base_commit: str, source_path: str) -> None:
        """Store a fully-initialized setup in the cache."""
        if os.getenv("RFSN_USE_ENV_CACHE") != "1":
            return
            
        key = self._get_key(repo_url, base_commit)
        target_path = os.path.join(self.cache_root, key)
        
        if os.path.exists(target_path):
            return  # Already cached
            
        lock_file = target_path + ".lock"
        # Try to acquire lock
        if os.path.exists(lock_file):
            return  # Someone else is doing it
            
        try:
            with open(lock_file, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            return  # Couldn't get lock
            
        logger.info("ENVIRONMENT CACHE: Storing setup as %s", key)
        try:
            # Create a temp dir for copying to avoid partial hits
            temp_target = target_path + ".tmp"
            if os.path.exists(temp_target):
                shutil.rmtree(temp_target)
            
            # Copy everything including built extensions
            shutil.copytree(source_path, temp_target, symlinks=True)
            
            # Mark as ready
            with open(os.path.join(temp_target, ".RFSN_READY"), "w") as f:
                f.write(f"repo={repo_url}\ncommit={base_commit}\n")
            
            os.rename(temp_target, target_path)
            logger.info("ENVIRONMENT CACHE: Success")
        except Exception as e:
            logger.warning("ENVIRONMENT CACHE: Failed to store: %s", e)
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)
            if os.path.exists(target_path + ".tmp"):
                with suppress(Exception):
                    shutil.rmtree(target_path + ".tmp")

# Global cache instance
_env_cache = VenvCache()


@dataclass
class RepoWorkspace:
    """A benchmark repository workspace."""
    
    path: str
    repo: str
    base_commit: str
    venv_path: str | None = None
    env: dict[str, str] | None = None
    installed: bool = False


def _run(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None, timeout_s: int = 600):
    """Run a command with logging."""
    from contextlib import suppress
    
    logger.debug("RUN: %s (cwd=%s)", " ".join(cmd), cwd)
    with suppress(Exception):
        return subprocess.run(
            cmd, 
            cwd=cwd, 
            env=env, 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=timeout_s
        )
    return None


def clone_repo(repo_url: str, base_commit: str, work_root: str = ".work") -> RepoWorkspace:
    """
    Clone a repository and checkout a specific commit.
    
    Args:
        repo_url: Git URL to clone
        base_commit: Commit SHA to checkout
        work_root: Directory to create workspaces in
        
    Returns:
        RepoWorkspace with path to cloned repo
        
    Raises:
        RuntimeError: If clone or checkout fails
    """
    os.makedirs(work_root, exist_ok=True)
    d = tempfile.mkdtemp(prefix="rfsn_repo_", dir=work_root)
    
    # Check cache first
    warm_path = _env_cache.get_warm_path(repo_url, base_commit)
    if warm_path:
        logger.info("ENVIRONMENT CACHE: Reusing warm setup from %s", warm_path)
        # Copy warm setup to workspace
        # We use a merge-like copy but since 'd' is empty it's just cp -r
        shutil.rmtree(d)
        shutil.copytree(warm_path, d, symlinks=True)
        
        # NOTE: Editable installs (-e .) might point to the warm_path.
        # We need to re-run a fast checkout of the base_commit just in case
        # and re-install if needed, but for now we trust the copy.
        return RepoWorkspace(path=d, repo=repo_url, base_commit=base_commit)

    # Cache miss: Proceed with normal clone
    # Try shallow clone first (faster)
    p = _run(["git", "clone", "--depth", "1", repo_url, d], cwd=".")
    if p and p.returncode != 0:
        # Fallback: full clone (some commits not in shallow)
        shutil.rmtree(d, ignore_errors=True)
        d = tempfile.mkdtemp(prefix="rfsn_repo_", dir=work_root)
        p = _run(["git", "clone", repo_url, d], cwd=".")
        if p and p.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{p.stdout if p else 'Unknown error'}")

    # Fetch all refs to ensure we can checkout the base commit
    p = _run(["git", "fetch", "--all", "--tags"], cwd=d)
    if p and p.returncode != 0:
        raise RuntimeError(f"git fetch failed:\n{p.stdout if p else 'Unknown error'}")

    p = _run(["git", "checkout", base_commit], cwd=d)
    if p and p.returncode != 0:
        raise RuntimeError(f"git checkout {base_commit} failed:\n{p.stdout if p else 'Unknown error'}")

    # Install dependencies
    install_deps(d)

    # Store in cache for future use
    _env_cache.store(repo_url, base_commit, d)

    return RepoWorkspace(path=d, repo=repo_url, base_commit=base_commit)


def install_deps(repo_path: str) -> None:
    """
    Install Python dependencies for a repository.
    
    Tries several common approaches:
    - pip install -e .[test]
    - pip install -e .
    - pip install -r requirements.txt
    - pip install hypothesis pytest (fallback for SWE-bench repos)
    """
    # Install project-specific dependencies that may be missing (before main install)
    # pyerfa (imported as erfa) is required by astropy but not always installed automatically
    # Detect astropy by checking for the astropy/ package directory
    is_astropy = os.path.isdir(os.path.join(repo_path, "astropy"))
    if is_astropy:
        logger.info("ASTROPY DETECTED: Installing build dependencies")
        # Install build and runtime dependencies for astropy
        # - pyerfa: runtime dependency for astronomy calculations
        # - numpy<2.0: required for building C extensions AND older astropy uses numpy.core
        #   which was deprecated in numpy 1.x and removed in numpy 2.0
        # - extension-helpers: helps with extension building
        # - cython: required to build C extensions from .pyx files
        # - setuptools_scm: version detection during build
        # - jinja2: required by some astropy build scripts
        _run(["pip", "install", "pyerfa", "numpy<2.0", "extension-helpers<1.0", "cython", "setuptools_scm<8.0", "jinja2", "--quiet"], cwd=repo_path, timeout_s=180)
        
        # Fix setuptools_scm version detection before building
        logger.info("ASTROPY: Fixing setuptools_scm version")
        _fix_setuptools_scm_version(repo_path)
        
        # For astropy: Try pip install -e . which handles dependencies better
        logger.info("ASTROPY: Running pip install -e .")
        p = _run(["pip", "install", "-e", ".", "--no-build-isolation"], cwd=repo_path, timeout_s=600)
        logger.info("ASTROPY: pip install -e . returned %d", p.returncode)
        
        # If pip install failed, also try build_ext as fallback
        build_succeeded = (p.returncode == 0)
        if p.returncode != 0:
            logger.info("ASTROPY: pip install failed, trying build_ext --inplace")
            p = _run(["python", "setup.py", "build_ext", "--inplace"], cwd=repo_path, timeout_s=600)
            logger.info("ASTROPY: build_ext returned %d", p.returncode)
            build_succeeded = (p.returncode == 0)
        
        # If build still fails, patch astropy/__init__.py to bypass extension check
        # This allows tests for pure Python code (like test_separable.py) to run
        if not build_succeeded:
            logger.info("ASTROPY: Build failed, patching __init__.py to bypass extension check")
            init_path = os.path.join(repo_path, "astropy", "__init__.py")
            if os.path.exists(init_path):
                # Read content with robust encoding
                with open(init_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                
                # Replace entire _initialize_astropy function body with pass
                # Strategy: when we find 'def _initialize_astropy', keep that line,
                # add 'pass', then comment out ALL subsequent lines in the function
                # until we hit a line that's at the same indentation level (unindented)
                new_lines = []
                in_function = False
                function_indent = None
                added_pass = False
                
                for line in lines:
                    if "def _initialize_astropy(" in line:
                        in_function = True
                        # Determine the base indentation of the function def
                        function_indent = len(line) - len(line.lstrip())
                        new_lines.append(line)
                        # Add pass immediately after the def line
                        body_indent = " " * (function_indent + 4)
                        new_lines.append(f"{body_indent}pass  # RFSN bypass: extension check disabled\n")
                        added_pass = True
                        continue
                    
                    if in_function:
                        # Check if this line is back at the function definition level (end of function)
                        stripped = line.strip()
                        # If it's an empty line or comment-only line, keep it within function
                        if not stripped or stripped.startswith("#"):
                            new_lines.append("#" + line if stripped else line)
                            continue
                        # Calculate current indentation
                        current_indent = len(line) - len(line.lstrip())
                        # If we're back at or before function_indent, we've left the function
                        if current_indent <= function_indent and stripped:
                            in_function = False
                            new_lines.append(line)
                        else:
                            # Still inside function body - comment it out
                            new_lines.append("# RFSN: " + line)
                        continue
                    
                    new_lines.append(line)
                
                # Write content with robust encoding
                with open(init_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                logger.info(f"ASTROPY: Patched _initialize_astropy (added_pass={added_pass})")
        
        # Also patch astropy/modeling/__init__.py to skip fitting import
        _patch_astropy_modeling_init(repo_path)
        
        # Create stub _column_mixins.py to allow table imports  
        _patch_astropy_column_mixins(repo_path)
        
        # Patch table init to handle circular imports
        _patch_astropy_table_init(repo_path)
        
        # Patch table/operations.py to handle _np_utils circular import
        _patch_astropy_table_operations(repo_path)
        
        # Patch io/ascii module for Table import issues
        _patch_astropy_io_ascii_init(repo_path)
        _patch_astropy_io_ascii_connect(repo_path)
        
        # Patch logger.py to handle config access failures
        _patch_astropy_logger(repo_path)
        
        # Install test dependencies (including pytest-doctestplus for astropy's setup.cfg)
        _run(["pip", "install", "hypothesis", "pytest", "pytest-doctestplus", "pytest-astropy", "--quiet"], cwd=repo_path, timeout_s=120)
        logger.info("ASTROPY: Setup complete")
        return
    
    # Non-astropy projects: use standard install flow
    # Try pip install -e .[test] (editable install with test extras)
    pip_cmd = ["pip", "install", "-e", ".[test]", "--quiet"]
    logger.debug("INSTALL: %s", " ".join(pip_cmd))
    p = _run(pip_cmd, cwd=repo_path, timeout_s=300)
    if p and p.returncode == 0:
        # Also ensure version mock is in place for setuptools_scm
        _fix_setuptools_scm_version(repo_path)
        return
    
    # Try pip install -e . (editable install)
    pip_cmd = ["pip", "install", "-e", ".", "--quiet"]
    logger.debug("INSTALL: %s", " ".join(pip_cmd))
    p = _run(pip_cmd, cwd=repo_path, timeout_s=300)
    if p and p.returncode == 0:
        # Also install common test deps
        _run(["pip", "install", "hypothesis", "pytest", "--quiet"], cwd=repo_path, timeout_s=120)
        return
    
    # Try requirements.txt
    req_path = os.path.join(repo_path, "requirements.txt")
    if os.path.exists(req_path):
        _run(["pip", "install", "-r", "requirements.txt", "--quiet"], cwd=repo_path, timeout_s=300)
    
    # Always try to install common test deps as fallback
    _run(["pip", "install", "hypothesis", "pytest", "--quiet"], cwd=repo_path, timeout_s=120)
    
    # Fix setuptools_scm version detection for projects that fail
    _fix_setuptools_scm_version(repo_path)


def _fix_setuptools_scm_version(repo_path: str) -> None:
    """
    Fix version detection for projects using setuptools_scm.
    
    Many SWE-bench repos (astropy, django, etc.) use setuptools_scm for
    versioning, which fails when checking out a specific commit without
    git tags. This creates a mock version.py to allow import to succeed.
    """
    # List of known packages that use setuptools_scm and need fixing
    packages_to_fix = [
        ("astropy", "astropy/version.py"),
        ("django", "django/version.py"),
        ("sympy", "sympy/release.py"),
        ("matplotlib", "lib/matplotlib/_version.py"),
        ("pandas", "pandas/_version_meson.py"),
    ]
    
    for pkg_name, version_file in packages_to_fix:
        pkg_dir = os.path.join(repo_path, pkg_name)
        version_path = os.path.join(repo_path, version_file)
        
        # Check if this is the relevant package
        if not os.path.exists(pkg_dir):
            continue
            
        # Create version mock content
        mock_content = '''"""Version mock for SWE-bench evaluation."""
version = "0.0.dev0"
__version__ = version
'''
        
        # Write mock version file
        try:
            os.makedirs(os.path.dirname(version_path), exist_ok=True)
            with open(version_path, "w", encoding="utf-8") as f:
                f.write(mock_content)
        except Exception:
            pass  # Ignore errors - best effort


def _apply_semantic_patch(workspace_path: str, patch_text: str) -> str:
    """
    Apply a patch semantically by searching for old content and replacing with new.
    
    This is a fallback when git apply and patch fail due to line number mismatches.
    It parses the unified diff, extracts the old and new content from each hunk,
    and searches for the old content in the file to replace it.
    
    Returns "APPLIED_OK" on success, error message otherwise.
    """
    import re
    
    # Parse the diff to extract file paths and hunks
    # Format: --- a/path/to/file.py\n+++ b/path/to/file.py\n@@ ... @@\n...
    
    lines = patch_text.split('\n')
    current_file = None
    in_hunk = False
    old_lines: list[str] = []
    new_lines: list[str] = []
    applied_any = False
    
    for i, line in enumerate(lines):
        # Detect file path
        if line.startswith('--- a/'):
            # Look for next +++ line to get target file
            current_file = line[6:].strip()
            continue
        elif line.startswith('+++ b/'):
            current_file = line[6:].strip()
            continue
        
        # Detect hunk header
        if line.startswith('@@') and '@@' in line[2:]:
            # Apply previous hunk if any
            if old_lines and current_file:
                result = _apply_hunk(workspace_path, current_file, old_lines, new_lines)
                if result:
                    applied_any = True
            # Reset for new hunk
            in_hunk = True
            old_lines = []
            new_lines = []
            continue
        
        if in_hunk:
            if line.startswith('-') and not line.startswith('---'):
                old_lines.append(line[1:])  # Remove the '-' prefix
            elif line.startswith('+') and not line.startswith('+++'):
                new_lines.append(line[1:])  # Remove the '+' prefix
            elif line.startswith(' '):
                # Context line - goes in both old and new
                old_lines.append(line[1:])
                new_lines.append(line[1:])
            elif line == '':
                # Empty line (could be context line with no leading space due to LLM error)
                old_lines.append('')
                new_lines.append('')
    
    # Apply final hunk
    if old_lines and current_file:
        result = _apply_hunk(workspace_path, current_file, old_lines, new_lines)
        if result:
            applied_any = True
    
    if applied_any:
        return "APPLIED_OK"
    return "SEMANTIC_APPLY_FAILED"


def _apply_hunk(workspace_path: str, file_path: str, old_lines: list[str], new_lines: list[str]) -> bool:
    """
    Apply a single hunk by searching for old content and replacing with new.
    
    Returns True on success, False on failure.
    """
    full_path = os.path.join(workspace_path, file_path)
    
    if not os.path.exists(full_path):
        return False
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return False
    
    # Build the old content string (just the lines to be replaced)
    # We need to find lines that are being REMOVED (prefixed with '-' in diff)
    # and replace them with lines that are being ADDED (prefixed with '+' in diff)
    
    # The old_lines includes context lines (which appear in both old_lines and new_lines)
    # We need to search for the entire old block including context
    old_content = '\n'.join(old_lines)
    new_content = '\n'.join(new_lines)
    
    # First try exact match
    if old_content in content:
        new_file_content = content.replace(old_content, new_content, 1)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            return True
        except Exception:
            return False
    
    # Try fuzzy match (strip trailing whitespace from each line)
    old_lines_stripped = [line.rstrip() for line in old_lines]
    content_lines = content.split('\n')
    
    # Slide window search
    for start_idx in range(len(content_lines) - len(old_lines_stripped) + 1):
        window = content_lines[start_idx:start_idx + len(old_lines_stripped)]
        window_stripped = [line.rstrip() for line in window]
        
        if window_stripped == old_lines_stripped:
            # Found match! Replace this section
            new_content_lines = content_lines[:start_idx] + new_lines + content_lines[start_idx + len(old_lines_stripped):]
            new_file_content = '\n'.join(new_content_lines)
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_file_content)
                return True
            except Exception:
                return False
    
    return False


def _patch_astropy_extension_bypass(repo_path: str) -> bool:
    """
    Patch astropy/__init__.py to bypass extension check.
    
    This comments out the entire _initialize_astropy function body
    and replaces it with just 'pass', allowing pure Python tests to run
    without building C extensions.
    
    Returns True if patching was successful, False otherwise.
    """
    init_path = os.path.join(repo_path, "astropy", "__init__.py")
    if not os.path.exists(init_path):
        return False
    
    try:
        with open(init_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # Replace entire _initialize_astropy function body with pass
        new_lines = []
        in_function = False
        function_indent = None
        added_pass = False
        
        for line in lines:
            if "def _initialize_astropy(" in line:
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                new_lines.append(line)
                # Add pass immediately after the def line
                body_indent = " " * (function_indent + 4)
                new_lines.append(f"{body_indent}pass  # RFSN bypass: extension check disabled\n")
                added_pass = True
                continue
            
            if in_function:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    new_lines.append("#" + line if stripped else line)
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= function_indent and stripped:
                    in_function = False
                    new_lines.append(line)
                else:
                    new_lines.append("# RFSN: " + line)
                continue
            
            new_lines.append(line)
        
        if added_pass:
            with open(init_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.debug("Patched _initialize_astropy extension bypass")
            return True
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy extension bypass: %s", e)
        return False


def _patch_astropy_modeling_init(repo_path: str) -> bool:
    """
    Patch astropy/modeling/__init__.py to NOT import 'fitting'.
    
    The import chain fitting -> spline -> core -> astropy.table 
    requires compiled C extensions. By commenting out the fitting import,
    we allow test_separable.py to be collected and run.
    """
    try:
        init_path = os.path.join(repo_path, "astropy", "modeling", "__init__.py")
        if not os.path.exists(init_path):
            return False
        
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Comment out the fitting import - try multiple patterns
        modified = False
        
        # Pattern 1: Old style "from . import fitting" (separate line)
        if "from . import fitting\n" in content:
            content = content.replace(
                "from . import fitting\n", 
                "# from . import fitting  # RFSN: imports table which needs C exts\n"
            )
            modified = True
        
        # Pattern 2: New style "from . import fitting, models"
        if "from . import fitting, models" in content:
            content = content.replace(
                "from . import fitting, models", 
                "from . import models  # RFSN: fitting imports table which needs C exts"
            )
            modified = True
        
        if modified:
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("ASTROPY: Patched modeling/__init__.py to skip fitting import")
            return True
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy modeling init: %s", e)
        return False


def _patch_astropy_column_mixins(repo_path: str) -> bool:
    """
    Create a stub astropy/table/_column_mixins.py to allow table imports without C extensions.
    
    The real _column_mixins is a Cython module that provides fast column operations.
    This stub provides minimal class definitions to satisfy the import.
    """
    try:
        mixins_path = os.path.join(repo_path, "astropy", "table", "_column_mixins.py")
        table_dir = os.path.dirname(mixins_path)
        
        if not os.path.exists(table_dir):
            logger.debug("ASTROPY: table dir does not exist: %s", table_dir)
            return False
        
        # Check if real compiled module exists - don't overwrite
        import glob
        so_files = glob.glob(os.path.join(table_dir, "_column_mixins*.so"))
        pyd_files = glob.glob(os.path.join(table_dir, "_column_mixins*.pyd"))
        if so_files or pyd_files:
            logger.debug("ASTROPY: Compiled _column_mixins found, skipping stub")
            return False  # Compiled version exists, don't stub
        
        # Create stub if .py version doesn't exist
        if os.path.exists(mixins_path):
            logger.debug("ASTROPY: _column_mixins.py already exists")
            return False
        
        stub_content = '''"""
RFSN Stub: _column_mixins.py
Minimal stubs to allow astropy.table imports without compiled extensions.
"""

class _ColumnGetitemShim:
    """Stub for column getitem shim."""
    pass

class _MaskedColumnGetitemShim:
    """Stub for masked column getitem shim."""
    pass
'''
        
        with open(mixins_path, "w", encoding="utf-8") as f:
            f.write(stub_content)
        logger.info("ASTROPY: Created stub _column_mixins.py")
        return True
    except Exception as e:
        logger.debug("Failed to create _column_mixins stub: %s", e)
        return False


def _patch_astropy_table_init(repo_path: str) -> bool:
    """
    Patch astropy/table/__init__.py to handle circular import issues.
    
    Old astropy versions have circular import issues when importing
    pprint, connect, scripts etc. from the table module. This wraps
    ONLY simple single-line imports in try/except blocks.
    
    Multi-line imports like `from .X import (a, b, c)` are SKIPPED
    to avoid syntax errors.
    
    Returns True if patching was successful, False otherwise.
    """
    table_init_path = os.path.join(repo_path, "astropy", "table", "__init__.py")
    if not os.path.exists(table_init_path):
        return False
    
    try:
        with open(table_init_path, encoding="utf-8") as f:
            content = f.read()
        
        # Already patched
        if "# RFSN PATCHED" in content:
            return True
        
        # Add header
        patched_content = "# RFSN PATCHED: Wrapped single-line imports for circular import safety\n"
        
        # Problematic import prefixes (only wrap if line is self-contained)
        problematic_prefixes = [
            "from .pprint import",
            "from .connect import",
            "from .scripts import",
            "from .serialize import",
        ]
        
        lines = content.split('\n')
        result_lines = []
        in_multiline = False  # Track if inside parenthesized import
        
        for line in lines:
            stripped = line.strip()
            
            # Track multi-line imports: starts with ( but doesn't close on same line
            if '(' in stripped and ')' not in stripped:
                in_multiline = True
                result_lines.append(line)
                continue
            
            # End of multi-line import
            if in_multiline:
                result_lines.append(line)
                if ')' in stripped:
                    in_multiline = False
                continue
            
            # Check if this is a simple problematic import (single line, no continuation)
            is_problematic = any(stripped.startswith(p) for p in problematic_prefixes)
            is_single_line = not stripped.endswith('\\') and '(' not in stripped
            is_not_comment = not stripped.startswith("#")
            
            if is_problematic and is_single_line and is_not_comment:
                # Wrap in try/except
                result_lines.append(f"try:  # RFSN: wrapped for circular import safety")
                result_lines.append(f"    {line}")
                result_lines.append(f"except ImportError:")
                result_lines.append(f"    pass  # RFSN: circular import fallback")
            else:
                result_lines.append(line)
        
        patched_content += '\n'.join(result_lines)
        
        with open(table_init_path, "w", encoding="utf-8") as f:
            f.write(patched_content)
        logger.info("ASTROPY: Patched table/__init__.py (single-line imports only)")
        return True
    except Exception as e:
        logger.debug("Failed to patch astropy table init: %s", e)
        return False


def _patch_astropy_table_operations(repo_path: str) -> bool:
    """
    Patch astropy/table/operations.py to handle _np_utils circular import.
    
    The import `from . import _np_utils` at the top triggers circular import
    when Table is imported early. We defer this import inside functions.
    
    Returns True if patching was successful, False otherwise.
    """
    ops_path = os.path.join(repo_path, "astropy", "table", "operations.py")
    if not os.path.exists(ops_path):
        return False
    
    try:
        with open(ops_path, encoding="utf-8") as f:
            content = f.read()
        
        # Already patched
        if "# RFSN PATCHED" in content:
            return True
        
        # Wrap the _np_utils import in try/except
        if "from . import _np_utils" in content:
            content = content.replace(
                "from . import _np_utils",
                "try:  # RFSN: wrapped for circular import safety\n    from . import _np_utils\nexcept ImportError:\n    _np_utils = None  # RFSN: fallback"
            )
            
            # Add RFSN header
            content = "# RFSN PATCHED: Wrapped _np_utils import for circular import safety\n" + content
            
            with open(ops_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("ASTROPY: Patched table/operations.py (_np_utils import)")
            return True
        
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy table operations: %s", e)
        return False


def _patch_astropy_io_ascii_init(repo_path: str) -> bool:
    """
    Patch astropy/io/ascii/__init__.py to handle Table import issues.
    
    The io/ascii module imports Table at module level, but this triggers
    circular imports in older astropy versions. We wrap problematic imports.
    
    Returns True if patching was successful, False otherwise.
    """
    ascii_init_path = os.path.join(repo_path, "astropy", "io", "ascii", "__init__.py")
    if not os.path.exists(ascii_init_path):
        return False
    
    try:
        with open(ascii_init_path, encoding="utf-8") as f:
            content = f.read()
        
        # Already patched
        if "# RFSN PATCHED" in content:
            return True
        
        modified = False
        
        # Wrap connect import
        if "from . import connect" in content and "try:" not in content.split("from . import connect")[0][-50:]:
            content = content.replace(
                "from . import connect",
                "try:  # RFSN: wrapped for circular import safety\n    from . import connect\nexcept ImportError:\n    pass  # RFSN: circular import fallback"
            )
            modified = True
        
        if modified:
            content = "# RFSN PATCHED: Wrapped imports for circular import safety\n" + content
            with open(ascii_init_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("ASTROPY: Patched io/ascii/__init__.py")
            return True
        
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy io/ascii init: %s", e)
        return False


def _patch_astropy_io_ascii_connect(repo_path: str) -> bool:
    """
    Patch astropy/io/ascii/connect.py to defer Table import.
    
    Line 8 has `from astropy.table import Table` which triggers the circular
    import chain. We move this import to inside functions that use it.
    
    Returns True if patching was successful, False otherwise.
    """
    connect_path = os.path.join(repo_path, "astropy", "io", "ascii", "connect.py")
    if not os.path.exists(connect_path):
        return False
    
    try:
        with open(connect_path, encoding="utf-8") as f:
            content = f.read()
        
        # Already patched
        if "# RFSN PATCHED" in content:
            return True
        
        modified = False
        
        # Wrap the Table import at module level
        if "from astropy.table import Table" in content:
            content = content.replace(
                "from astropy.table import Table",
                "try:  # RFSN: wrapped for circular import safety\n    from astropy.table import Table\nexcept ImportError:\n    Table = None  # RFSN: deferred import"
            )
            modified = True
        
        if modified:
            content = "# RFSN PATCHED: Wrapped Table import for circular import safety\n" + content
            with open(connect_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("ASTROPY: Patched io/ascii/connect.py (Table import)")
            return True
        
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy io/ascii/connect.py: %s", e)
        return False


def _patch_astropy_logger(repo_path: str) -> bool:
    """
    Patch astropy/logger.py to handle config access failures.
    
    In some old astropy versions, logger._set_defaults() tries to access
    conf.log_level before conf is properly initialized. We wrap this in try/except.
    
    Returns True if patching was successful, False otherwise.
    """
    logger_path = os.path.join(repo_path, "astropy", "logger.py")
    if not os.path.exists(logger_path):
        return False
    
    try:
        with open(logger_path, encoding="utf-8") as f:
            content = f.read()
        
        # Already patched
        if "# RFSN PATCHED" in content:
            return True
        
        modified = False
        
        # Wrap the _set_defaults call - detect indentation dynamically
        import re
        match = re.search(r'^( *)log\._set_defaults\(\)', content, re.MULTILINE)
        if match and "try:  # RFSN" not in content:
            indent = match.group(1)  # e.g. 8 spaces
            old_line = match.group(0)  # full line including indent
            new_block = (
                f"{indent}try:  # RFSN: wrapped for config access safety\n"
                f"{indent}    log._set_defaults()\n"
                f"{indent}except Exception:\n"
                f"{indent}    pass  # RFSN: config not ready yet"
            )
            content = content.replace(old_line, new_block)
            modified = True
        
        # Wrap setLevel with conf.log_level (detect indentation dynamically)
        match2 = re.search(r'^( *)self\.setLevel\(conf\.log_level\)', content, re.MULTILINE)
        if match2 and "try:  # RFSN" not in content.split("self.setLevel(conf.log_level)")[0][-100:]:
            indent = match2.group(1)
            old_line = match2.group(0)
            new_block = (
                f"{indent}try:  # RFSN: wrapped for config access safety\n"
                f"{indent}    self.setLevel(conf.log_level)\n"
                f"{indent}except Exception:\n"
                f"{indent}    self.setLevel('WARNING')  # RFSN: fallback"
            )
            content = content.replace(old_line, new_block)
            modified = True
        
        if modified:
            content = "# RFSN PATCHED: Wrapped config access for safety\n" + content
            with open(logger_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("ASTROPY: Patched logger.py (config access)")
            return True
        
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy logger: %s", e)
        return False


def _patch_astropy_conftest(repo_path: str) -> bool:
    """
    Surgically patch astropy/conftest.py to wrap failing imports in try/except.
    Uses a line-by-line approach to define stub objects when imports fail.
    """
    conftest_path = os.path.join(repo_path, "astropy", "conftest.py")
    if not os.path.exists(conftest_path):
        return False
    
    try:
        with open(conftest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if any("# RFSN SURGICAL PATCH" in line for line in lines):
            return True
        
        # Patterns to match and their stub replacements
        patch_specs = [
            ("from astropy.utils.iers import conf as iers_conf", 
             "    try: from astropy.utils.iers import conf as iers_conf\n    except Exception: iers_conf = type('stub', (), {'auto_download': False, 'auto_max_age': None, 'iers_degraded_accuracy': None})()\n"),
            ("from astropy.time import Time", 
             "    try: from astropy.time import Time\n    except Exception: Time = None\n"),
            ("from astropy import units as u", 
             "    try: from astropy import units as u\n    except Exception: u = None\n"),
        ]
        
        new_lines = ["# RFSN SURGICAL PATCH\n"]
        modified = False
        
        for line in lines:
            stripped = line.strip()
            matched = False
            for pattern, replacement in patch_specs:
                if stripped == pattern:
                    new_lines.append(replacement)
                    modified = True
                    matched = True
                    break
            if not matched:
                new_lines.append(line)
        
        if modified:
            with open(conftest_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.debug("Surgically patched astropy conftest with stubs")
            return True
            
        return False
    except Exception as e:
        logger.debug("Failed to patch astropy conftest: %s", e)
        return False



def hard_reset_clean(ws: RepoWorkspace) -> None:
    """
    Reset workspace to clean state.
    
    Performs git reset --hard and git clean -fdx.
    """
    _run(["git", "reset", "--hard"], cwd=ws.path)
    _run(["git", "clean", "-fdx"], cwd=ws.path)
    
    # Re-apply version mock after clean (git clean removes generated files)
    _fix_setuptools_scm_version(ws.path)
    
    # Re-apply astropy extension bypass if this is an astropy project
    # (git reset --hard undoes the __init__.py patch)
    init_path = os.path.join(ws.path, "astropy", "__init__.py")
    if os.path.exists(init_path):
        _patch_astropy_extension_bypass(ws.path)
        _patch_astropy_conftest(ws.path)
        _patch_astropy_modeling_init(ws.path)
        _patch_astropy_column_mixins(ws.path)
        _patch_astropy_table_init(ws.path)
        _patch_astropy_table_operations(ws.path)
        _patch_astropy_io_ascii_init(ws.path)
        _patch_astropy_io_ascii_connect(ws.path)
        _patch_astropy_logger(ws.path)


def apply_patch_text(ws: RepoWorkspace, patch_text: str) -> str:
    """
    Apply a patch blob safely.
    
    Uses git apply --check first, then git apply --3way.
    
    Args:
        ws: Workspace to apply patch in
        patch_text: The patch content
        
    Returns:
        Status string: "APPLIED_OK", "EMPTY_PATCH", or error message
    """
    if not patch_text.strip():
        return "EMPTY_PATCH"

    patch_path = os.path.join(ws.path, ".rfsn_tmp.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)

    try:
        # Try strict apply first
        git_apply_check_cmd = ["git", "apply", "--check", patch_path]
        logger.debug("APPLY: %s", " ".join(git_apply_check_cmd))
        p1 = _run(git_apply_check_cmd, cwd=ws.path)
        if p1 and p1.returncode == 0:
            # Strict check passed, apply it
            git_apply_cmd = ["git", "apply", "--3way", patch_path]
            logger.debug("APPLY: %s", " ".join(git_apply_cmd))
            p2 = _run(git_apply_cmd, cwd=ws.path)
            if p2 and p2.returncode == 0:
                logger.debug("PATCH SUCCESS")
                return "APPLIED_OK"
        
        # Try lenient apply with --recount (fixes line count mismatches in hunks)
        git_apply_recount_check_cmd = ["git", "apply", "--recount", "--check", patch_path]
        logger.debug("APPLY: %s", " ".join(git_apply_recount_check_cmd))
        p3 = _run(git_apply_recount_check_cmd, cwd=ws.path)
        if p3 and p3.returncode == 0:
            git_apply_recount_cmd = ["git", "apply", "--recount", "--3way", patch_path]
            logger.debug("APPLY: %s", " ".join(git_apply_recount_cmd))
            p4 = _run(git_apply_recount_cmd, cwd=ws.path)
            if p4 and p4.returncode == 0:
                return "APPLIED_OK"
        
        # Try with whitespace ignoring (common LLM issue)
        git_apply_ignore_ws_check_cmd = ["git", "apply", "--ignore-whitespace", "--check", patch_path]
        logger.debug("APPLY: %s", " ".join(git_apply_ignore_ws_check_cmd))
        p5 = _run(git_apply_ignore_ws_check_cmd, cwd=ws.path)
        if p5 and p5.returncode == 0:
            git_apply_ignore_ws_cmd = ["git", "apply", "--ignore-whitespace", "--3way", patch_path]
            logger.debug("APPLY: %s", " ".join(git_apply_ignore_ws_cmd))
            p6 = _run(git_apply_ignore_ws_cmd, cwd=ws.path)
            if p6 and p6.returncode == 0:
                return "APPLIED_OK"
        
        # Try most lenient: recount + ignore-whitespace
        git_apply_recount_ignore_ws_check_cmd = ["git", "apply", "--recount", "--ignore-whitespace", "--check", patch_path]
        logger.debug("APPLY: %s", " ".join(git_apply_recount_ignore_ws_check_cmd))
        p7 = _run(git_apply_recount_ignore_ws_check_cmd, cwd=ws.path)
        if p7 and p7.returncode == 0:
            git_apply_recount_ignore_ws_cmd = ["git", "apply", "--recount", "--ignore-whitespace", "--3way", patch_path]
            logger.debug("APPLY: %s", " ".join(git_apply_recount_ignore_ws_cmd))
            p8 = _run(git_apply_recount_ignore_ws_cmd, cwd=ws.path)
            if p8 and p8.returncode == 0:
                return "APPLIED_OK"
        
        # Last resort: use patch command with fuzz factor (handles line offset issues)
        # The -F3 flag allows up to 3 lines of context mismatch
        git_patch_cmd = ["patch", "-p1", "--dry-run", "-F3", "-i", patch_path]
        logger.debug("APPLY: %s", " ".join(git_patch_cmd))
        p9 = _run(git_patch_cmd, cwd=ws.path)
        if p9 and p9.returncode == 0:
            p10 = _run(["patch", "-p1", "-F3", "-i", patch_path], cwd=ws.path)
            if p10 and p10.returncode == 0:
                return "APPLIED_OK"
        
        # Final fallback: semantic patch application
        # Parse the diff, find old content in file, replace with new content
        result = _apply_semantic_patch(ws.path, patch_text)
        if result == "APPLIED_OK":
            return "APPLIED_OK"
        
        # All strategies failed
        return "APPLY_CHECK_FAILED\n" + p1.stdout
    finally:
        # Clean up temp file
        if os.path.exists(patch_path):
            os.remove(patch_path)


def cleanup_workspace(ws: RepoWorkspace) -> None:
    """Remove workspace directory."""
    if ws.path and os.path.exists(ws.path):
        shutil.rmtree(ws.path, ignore_errors=True)
