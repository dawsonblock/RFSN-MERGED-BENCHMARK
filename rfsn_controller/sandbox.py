"""Sandbox utilities for the RFSN controller.

This module defines a simple abstraction for creating and managing a disposable
git sandbox. All filesystem and git operations are executed within this
sandbox to isolate side effects. Only public GitHub repositories are
allowed to be cloned.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

from .command_allowlist import is_command_allowed


@dataclass
class DockerResult:
    """Result from a Docker command execution."""

    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class Sandbox:
    """A disposable workspace for running git operations."""

    root: str  # root directory of the sandbox
    repo_dir: str  # path to the cloned repository within the sandbox
    worktree_counter: int = 0
    allowed_commands: set[str] | None = None  # Language-specific command allowlist


_SANDBOX_COUNTER = count(1)
_WORKTREE_COUNTER_LOCK = threading.Lock()


def _resolve_path(sb: Sandbox, path: str) -> str:
    """Resolve and validate a path to ensure it's within the sandbox repo.
    
    Args:
        sb: The sandbox instance.
        path: Relative path to resolve.
        
    Returns:
        Absolute resolved path string.
        
    Raises:
        ValueError: If the path escapes the sandbox root.
    """
    # Base is the repo directory
    base = Path(sb.repo_dir).resolve()
    # Resolve the full target path
    # Handle absolute paths that are actually inside the repo (e.g. from previous resolves)
    if os.path.isabs(path):
        target = Path(path).resolve()
    else:
        target = (base / path).resolve()
        
    # Check if target is relative to base
    if not target.is_relative_to(base):
         raise ValueError(f"Security Violation: Path traversal attempt blocked: {path}")
    return str(target)


def _run(
    cmd: str, cwd: str, timeout_sec: int = 120, allowed_commands: set[str] | None = None
) -> tuple[int, str, str]:
    """Run a shell command and capture its output.

    Args:
        cmd: The command to run.
        cwd: Working directory.
        timeout_sec: Timeout for the command.
        allowed_commands: Optional set of allowed command names. If provided, only these commands are allowed.

    Returns:
        A tuple of (exit_code, stdout, stderr).
    """
    # Check if command is allowed by global security policy
    is_allowed, reason = is_command_allowed(cmd)
    if not is_allowed:
        return 1, "", f"Command blocked by security policy: {reason}"

    # Parse command into list (no shell=True for security)
    # Use shlex.split to handle quoted arguments properly
    try:
        cmd_list = shlex.split(cmd)
    except ValueError as e:
        return 1, "", f"Command parsing error: {e}"

    # Check against language-specific allowlist if provided
    if allowed_commands is not None and cmd_list:
        base_cmd = os.path.basename(cmd_list[0])
        if base_cmd not in allowed_commands:
            allowed_list = sorted(allowed_commands)
            preview = ", ".join(allowed_list[:10])
            extra_count = max(len(allowed_list) - 10, 0)
            extra_info = ""
            if extra_count > 0:
                extra_info = (
                    f" and {extra_count} more. See the project documentation for the full list of allowed commands."
                )
            return (
                1,
                "",
                f"Command '{base_cmd}' is not allowed for this project type. "
                f"Here are some allowed commands: {preview}{extra_info}",
            )

    # Track subprocess call in budget (if budget is configured)
    try:
        from .budget import record_subprocess_call_global
        record_subprocess_call_global()
    except ImportError:
        pass  # Budget module not available

    # Remove secret API keys from the environment to prevent accidental leakage to subprocesses
    # Env Allowlist: Only pass safe variables to the subprocess
    # Prevents sensitive credentials (AWS, KUBE, etc.) from leaking
    ALLOWED_ENV_VARS = {"PATH", "HOME", "LANG", "PYTHONPATH"}
    safe_env = {
        k: v for k, v in os.environ.items() if k in ALLOWED_ENV_VARS
    }
    p = subprocess.run(
        cmd_list,
        cwd=cwd,
        shell=False,  # Explicitly disable shell
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        env=safe_env, check=False,
    )
    return p.returncode, p.stdout, p.stderr


def create_sandbox(*, run_id: str | None = None) -> Sandbox:
    """Create a new disposable sandbox directory.

    Returns:
        A Sandbox object with paths configured.
    """
    if run_id:
        root = os.path.join(tempfile.gettempdir(), f"rfsn_sb_{run_id}")
    else:
        suffix = next(_SANDBOX_COUNTER)
        root = os.path.join(tempfile.gettempdir(), f"rfsn_sb_{suffix:06d}")
    os.makedirs(root, exist_ok=True)
    repo_dir = os.path.join(root, "repo")
    return Sandbox(root=root, repo_dir=repo_dir)


def destroy_sandbox(sb: Sandbox) -> None:
    """Recursively delete the sandbox directory."""
    if os.path.exists(sb.root):
        shutil.rmtree(sb.root, ignore_errors=True)


def clone_public_github(sb: Sandbox, github_url: str) -> dict[str, Any]:
    """Clone a public GitHub repository into the sandbox.

    This enforces that only public GitHub URLs are accepted and that no
    credentials or query parameters are allowed. If the repo is already
    cloned, it returns a note instead of re-cloning.

    Args:
        sb: The sandbox into which the repo should be cloned.
        github_url: The public GitHub URL of the repository.

    Returns:
        A dictionary indicating success and any stdout/stderr.
    """
    # Support local paths
    if os.path.exists(github_url) or github_url.startswith(("/", "./", "../")):
        try:
            # Clean up destination if it exists
            if os.path.exists(sb.repo_dir):
                shutil.rmtree(sb.repo_dir)
            
            # Copy source to repo_dir
            # If github_url is relative, make it absolute or respect cwd
            src_path = os.path.abspath(github_url)
            shutil.copytree(src_path, sb.repo_dir)
            
            # Initialize git if not present
            if not os.path.exists(os.path.join(sb.repo_dir, ".git")):
                # Git init
                c1, o1, e1 = _run("git init", cwd=sb.repo_dir)
                if c1 != 0:
                    return {"ok": False, "error": f"git init failed: {e1}"}
                
                # Git add
                c2, o2, e2 = _run("git add .", cwd=sb.repo_dir)
                if c2 != 0:
                    return {"ok": False, "error": f"git add failed: {e2}"}
                
                # Git commit
                c3, o3, e3 = _run("git commit -m 'Initial commit from local source'", cwd=sb.repo_dir)
                if c3 != 0:
                    return {"ok": False, "error": f"git commit failed: {e3}"}
                
            return {"ok": True, "note": f"Cloned from local source: {src_path}"}
        except Exception as e:
            return {"ok": False, "error": f"Local clone failed: {e!s}"}

    # public-only enforcement
    if not github_url.startswith("https://github.com/"):
        return {
            "ok": False,
            "error": "Only public GitHub https://github.com/OWNER/REPO URLs allowed.",
        }
    # Block credentials but allow valid query params for GitHub
    if "@" in github_url or "token" in github_url.lower():
        return {"ok": False, "error": "No credentials allowed."}
    # Block query params that might contain tokens
    parsed = github_url.split("?")
    if len(parsed) > 1:
        query_part = parsed[1].lower()
        if "token" in query_part or "access_token" in query_part or "password" in query_part:
            return {"ok": False, "error": "No credentials allowed in query params."}

    os.makedirs(sb.repo_dir, exist_ok=True)
    # clone into empty dir
    if os.path.exists(os.path.join(sb.repo_dir, ".git")):
        return {"ok": True, "note": "Repo already cloned."}

    parent = os.path.dirname(sb.repo_dir)
    code, out, err = _run(
        f"git clone {github_url} {sb.repo_dir}",
        cwd=parent,
        timeout_sec=600,
        allowed_commands=sb.allowed_commands,
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def checkout(sb: Sandbox, ref: str) -> dict[str, Any]:
    """Check out a specific git ref inside the sandboxed repository."""
    code, out, err = _run(
        f"git checkout {ref}",
        cwd=sb.repo_dir,
        timeout_sec=120,
        allowed_commands=sb.allowed_commands,
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def git_status(sb: Sandbox) -> dict[str, Any]:
    """Get a porcelain status of the repository."""
    code, out, err = _run(
        "git status --porcelain=v1",
        cwd=sb.repo_dir,
        timeout_sec=60,
        allowed_commands=sb.allowed_commands,
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def reset_hard(sb: Sandbox) -> dict[str, Any]:
    """Reset any changes and clean untracked files in the repository."""
    c1, o1, e1 = _run(
        "git reset --hard", cwd=sb.repo_dir, timeout_sec=120, allowed_commands=sb.allowed_commands
    )
    c2, o2, e2 = _run(
        "git clean -fd", cwd=sb.repo_dir, timeout_sec=120, allowed_commands=sb.allowed_commands
    )
    ok = c1 == 0 and c2 == 0
    return {"ok": ok, "stdout": o1 + o2, "stderr": e1 + e2}


# Cache for expensive operations
_tree_cache: dict[str, tuple[int, list[str]]] = {}
_file_cache: dict[str, tuple[int, str]] = {}
_cache_ttl_steps = 60
_cache_epoch = 0


def _tick_cache_epoch() -> int:
    global _cache_epoch
    _cache_epoch += 1
    return int(_cache_epoch)


def list_tree(sb: Sandbox, max_files: int = 400, use_cache: bool = True) -> dict[str, Any]:
    """Return a flattened list of files in the repository, pruning junk directories.

    Args:
        sb: The sandbox instance.
        max_files: Maximum number of files to return.
        use_cache: Whether to use cached results if available.

    Returns:
        Dictionary with ok status and files list.
    """
    cache_key = f"{sb.repo_dir}:{max_files}"
    now_step = _tick_cache_epoch()

    # Check cache
    if use_cache and cache_key in _tree_cache:
        cached_step, cached_files = _tree_cache[cache_key]
        if (now_step - int(cached_step)) < int(_cache_ttl_steps):
            return {"ok": True, "files": cached_files[:max_files]}

    found_files: list[str] = []
    for root, dirs, fnames in os.walk(sb.repo_dir):
        rel_root = os.path.relpath(root, sb.repo_dir).replace("\\", "/")
        if rel_root.startswith(".git"):
            continue
        # prune common junk
        dirs[:] = [
            d
            for d in dirs
            if d
            not in [
                ".git",
                "node_modules",
                ".venv",
                "venv",
                "__pycache__",
                "dist",
                "build",
                ".next",
                "out",
                "target",
                "target",
            ]
        ]
        for f in fnames:
            rel = os.path.normpath(os.path.join(rel_root, f)).replace("\\", "/")
            if rel.startswith(".git/"):
                continue
            found_files.append(rel.lstrip("./"))
            if len(found_files) >= max_files:
                found_files.sort()
                _tree_cache[cache_key] = (now_step, found_files)
                return {"ok": True, "files": found_files}
    found_files.sort()
    _tree_cache[cache_key] = (now_step, found_files)
    return {"ok": True, "files": found_files}


def read_file(
    sb: Sandbox, path: str, max_bytes: int = 120_000, use_cache: bool = True
) -> dict[str, Any]:
    """Read a file from the repository, returning its text truncated to max_bytes.

    Args:
        sb: The sandbox instance.
        path: Path to the file (relative to repo root).
        max_bytes: Maximum bytes to read.
        use_cache: Whether to use cached results if available.

    Returns:
        Dictionary with ok status, content, and path.
    """

    try:
        full_path = _resolve_path(sb, path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    
    # Try smart file cache first (better LRU eviction)
    if use_cache:
        try:
            from .smart_file_cache import get_file_cache
            smart_cache = get_file_cache()
            cached = smart_cache.get(full_path)
            if cached is not None:
                content = cached[:max_bytes] if len(cached) > max_bytes else cached
                return {"ok": True, "content": content, "path": path}
        except ImportError:
            pass  # Smart cache not available, use legacy
    
    # Legacy cache fallback
    cache_key = f"{full_path}:{max_bytes}"
    now_step = _tick_cache_epoch()

    if use_cache and cache_key in _file_cache:
        cached_step, content = _file_cache[cache_key]
        if (now_step - int(cached_step)) < int(_cache_ttl_steps):
            return {"ok": True, "content": content, "path": path}

    if not os.path.exists(full_path):
        return {"ok": False, "error": f"File not found: {path}"}
    try:
        with open(full_path, encoding="utf-8", errors="ignore") as f:
            content = f.read(max_bytes)
        
        # Store in both caches
        _file_cache[cache_key] = (now_step, content)
        if use_cache:
            try:
                from .smart_file_cache import get_file_cache
                get_file_cache().put(full_path, content)
            except ImportError:
                pass
        
        return {"ok": True, "content": content, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def pip_install(sb: Sandbox, packages: str, timeout_sec: int = 300) -> dict[str, Any]:
    """Install Python packages using pip in the sandboxed repository.

    Args:
        sb: The sandbox instance.
        packages: Space-separated list of packages to install (e.g., "requests numpy").
        timeout_sec: Maximum time to wait for installation.

    Returns:
        Result dictionary with ok status, stdout, and stderr.
    """
    # Use venv pip if available, otherwise system pip
    venv_pip = os.path.join(sb.repo_dir, ".venv", "bin", "pip")
    pip_cmd = venv_pip if os.path.exists(venv_pip) else "pip"
    cmd = f"{pip_cmd} install {packages}"
    code, out, err = _run(
        cmd, cwd=sb.repo_dir, timeout_sec=timeout_sec, allowed_commands=sb.allowed_commands
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def pip_install_requirements(
    sb: Sandbox, requirements_file: str = "requirements.txt", timeout_sec: int = 300
) -> dict[str, Any]:
    """Install packages from a requirements.txt file.

    Args:
        sb: The sandbox instance.
        requirements_file: Path to requirements.txt file (relative to repo root).
        timeout_sec: Maximum time to wait for installation.

    Returns:
        Result dictionary with ok status, stdout, and stderr.
    """
    full_path = os.path.join(sb.repo_dir, requirements_file)
    if not os.path.exists(full_path):
        return {"ok": False, "error": f"Requirements file not found: {requirements_file}"}
    # Use venv pip if available, otherwise system pip
    venv_pip = os.path.join(sb.repo_dir, ".venv", "bin", "pip")
    pip_cmd = venv_pip if os.path.exists(venv_pip) else "pip"
    cmd = f"{pip_cmd} install -r {requirements_file}"
    code, out, err = _run(
        cmd, cwd=sb.repo_dir, timeout_sec=timeout_sec, allowed_commands=sb.allowed_commands
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def create_venv(sb: Sandbox, venv_path: str = ".venv", timeout_sec: int = 60) -> dict[str, Any]:
    """Create a Python virtual environment in the sandbox.

    Args:
        sb: The sandbox instance.
        venv_path: Path for the virtual environment (relative to repo root).
        timeout_sec: Maximum time to wait for creation.

    Returns:
        Result dictionary with ok status, stdout, and stderr.
    """
    full_path = os.path.join(sb.repo_dir, venv_path)
    if os.path.exists(full_path):
        return {"ok": True, "note": f"Virtual environment already exists at {venv_path}"}
    cmd = f"python -m venv {venv_path}"
    code, out, err = _run(
        cmd, cwd=sb.repo_dir, timeout_sec=timeout_sec, allowed_commands=sb.allowed_commands
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def pip_install_progressive(sb: Sandbox, packages: str, timeout_sec: int = 300) -> dict[str, Any]:
    """Install Python packages one at a time, continuing on failures.

    This progressive installation strategy allows partial success when some
    packages are unavailable or fail to install.

    Args:
        sb: The sandbox instance.
        packages: Space-separated list of packages to install.
        timeout_sec: Maximum time per package installation.

    Returns:
        Result with overall status, successful packages, failed packages,
        and detailed results for each package.
    """
    # Use venv pip if available, otherwise system pip
    venv_pip = os.path.join(sb.repo_dir, ".venv", "bin", "pip")
    pip_cmd = venv_pip if os.path.exists(venv_pip) else "pip"

    package_list = packages.split()
    results = []
    successful = []
    failed = []

    for pkg in package_list:
        cmd = f"{pip_cmd} install {pkg}"
        code, out, err = _run(
            cmd, cwd=sb.repo_dir, timeout_sec=timeout_sec, allowed_commands=sb.allowed_commands
        )
        pkg_result = {
            "package": pkg,
            "ok": code == 0,
            "exit_code": code,
            "stdout": out,
            "stderr": err,
        }
        results.append(pkg_result)
        if code == 0:
            successful.append(pkg)
        else:
            failed.append(pkg)

    return {
        "ok": len(failed) == 0,
        "successful_packages": successful,
        "failed_packages": failed,
        "total_packages": len(package_list),
        "results": results,
    }


def find_local_module(sb: Sandbox, module_name: str) -> dict[str, Any]:
    """Search for a local module in the repository.

    This helps identify when a "missing" import is actually a local module
    that needs to be added to PYTHONPATH.

    Args:
        sb: The sandbox instance.
        module_name: Name of the module to search for.

    Returns:
        Result with found paths and PYTHONPATH suggestion.
    """
    module_variations = [
        module_name,
        module_name.replace("-", "_"),
        module_name.replace("_", "-"),
        f"{module_name}.py",
        f"{module_name}/__init__.py",
    ]

    found_paths = []
    for variation in module_variations:
        code, out, err = _run(
            f"find . -name '{variation}' -type f 2>/dev/null",
            cwd=sb.repo_dir,
            timeout_sec=30,
            allowed_commands=sb.allowed_commands,
        )
        if code == 0 and out.strip():
            for line in out.strip().splitlines():
                if line and not line.startswith("."):
                    found_paths.append(line.lstrip("./"))

    return {
        "ok": len(found_paths) > 0,
        "module_name": module_name,
        "found_paths": found_paths,
        "pythonpath_suggestion": f"export PYTHONPATH={sb.repo_dir}:$PYTHONPATH"
        if found_paths
        else None,
    }


def set_pythonpath(sb: Sandbox, path: str = "") -> dict[str, Any]:
    """Set PYTHONPATH for the sandbox environment.

    Note: This sets PYTHONPATH for subsequent commands in the same session.

    Args:
        sb: The sandbox instance.
        path: Path to add to PYTHONPATH (default: repo root).

    Returns:
        Result with status and the PYTHONPATH value.
    """
    if not path:
        path = sb.repo_dir
    cmd = f"export PYTHONPATH={path}:$PYTHONPATH && echo $PYTHONPATH"
    code, out, err = _run(
        cmd, cwd=sb.repo_dir, timeout_sec=10, allowed_commands=sb.allowed_commands
    )
    return {
        "ok": code == 0,
        "pythonpath": out.strip() if code == 0 else path,
        "stdout": out,
        "stderr": err,
    }


def grep(sb: Sandbox, query: str, max_matches: int = 200) -> dict[str, Any]:
    """Search recursively for a text query in the repository using grep."""
    query = query.replace("\n", " ")
    # Escape single quotes in query for shell safety
    query_escaped = query.replace("'", "'\\''")
    code, out, err = _run(
        f"grep -R --line-number '{query_escaped}' .",
        cwd=sb.repo_dir,
        timeout_sec=60,
        allowed_commands=sb.allowed_commands,
    )
    lines = (out + err).splitlines()[:max_matches]
    return {"ok": True, "matches": lines}


def run_cmd(sb: Sandbox, cmd: str, timeout_sec: int = 120) -> dict[str, Any]:
    """Run an arbitrary shell command inside the repository."""
    code, out, err = _run(
        cmd, cwd=sb.repo_dir, timeout_sec=timeout_sec, allowed_commands=sb.allowed_commands
    )
    return {"ok": code == 0, "exit_code": code, "stdout": out, "stderr": err}


def apply_patch(sb: Sandbox, diff: str) -> dict[str, Any]:
    """Apply a unified diff directly to the repository."""
    return apply_patch_in_dir(sb.repo_dir, diff)


def make_worktree(sb: Sandbox, *, suffix: str | None = None) -> str:
    """Create a detached worktree for testing candidate patches."""
    if suffix is None:
        with _WORKTREE_COUNTER_LOCK:
            sb.worktree_counter += 1
            suffix = f"{sb.worktree_counter:06d}"
    wt = os.path.join(sb.root, f"wt_{suffix}")
    # Escape path for shell safety
    wt_escaped = shlex.quote(wt)
    code, out, err = _run(
        f"git worktree add --detach {wt_escaped}",
        cwd=sb.repo_dir,
        timeout_sec=60,
        allowed_commands=sb.allowed_commands,
    )
    if code != 0:
        raise RuntimeError(f"worktree add failed: {err}\n{out}")
    return wt


def drop_worktree(sb: Sandbox, wt_dir: str) -> None:
    """Remove a detached worktree."""
    wt_escaped = shlex.quote(wt_dir)
    _run(
        f"git worktree remove --force {wt_escaped}",
        cwd=sb.repo_dir,
        timeout_sec=60,
        allowed_commands=sb.allowed_commands,
    )
    if os.path.exists(wt_dir):
        shutil.rmtree(wt_dir, ignore_errors=True)


def apply_patch_in_dir(wt_dir: str, diff: str) -> dict[str, Any]:
    """Apply a unified diff inside a specific worktree."""
    # Use subprocess with list instead of shell=True for security
    p = subprocess.run(
        ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "-"],
        cwd=wt_dir,
        shell=False,
        text=True,
        input=diff,
        capture_output=True,
        timeout=60, check=False,
    )
    if p.returncode == 0:
        return {
            "ok": True,
            "exit_code": 0,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }

    # Fallback to `patch` utility which is more lenient
    p2 = subprocess.run(
        ["patch", "-p1", "--batch", "--forward", "--ignore-whitespace"],
        cwd=wt_dir,
        shell=False,
        text=True,
        input=diff,
        capture_output=True,
        timeout=60, check=False,
    )
    return {
        "ok": p2.returncode == 0,
        "exit_code": p2.returncode,
        "stdout": p.stdout + "\n--- fallback ---\n" + p2.stdout,
        "stderr": p.stderr + "\n--- fallback ---\n" + p2.stderr,
    }


def docker_run(
    sb: Sandbox,
    cmd: str | list[str],
    timeout_sec: int = 120,
    network: bool = True,
    docker_image: str = "python:3.11-slim",
    cpu: float = 2.0,
    mem_mb: int = 4096,
    pids: int = 256,
    read_only: bool = False,
    use_cache: bool = True,
    use_warm_pool: bool = False,
) -> DockerResult:
    """Run a command inside a Docker container with the repo mounted.

    Args:
        sb: The sandbox containing the repo.
        cmd: The command to run inside the container.
        timeout_sec: Timeout for the command.
        network: Whether to enable network (True) or disable (False).
        docker_image: Docker image to use.
        cpu: CPU limit (e.g., 2.0 for 2 CPUs).
        mem_mb: Memory limit in MB.
        pids: Process ID limit.
        read_only: Whether to mount repo as read-only with /tmp as tmpfs.
        use_cache: Whether to use cache volumes for faster dependency installs.
        use_warm_pool: Use pre-warmed containers for faster execution (2-5s savings).

    Returns:
        DockerResult with execution status and output.
    """
    # Normalize command into argv to avoid shell execution
    import shlex
    if isinstance(cmd, str):
        cmd_argv = shlex.split(cmd)
    else:
        cmd_argv = list(cmd)
    if not cmd_argv:
        return DockerResult(ok=False, exit_code=2, stdout='', stderr='empty command', timed_out=False)

    # Try warm pool first for faster execution
    if use_warm_pool and not read_only:
        try:
            from .docker_pool import get_warm_pool
            pool = get_warm_pool()
            container = pool.get_or_create(docker_image, sb.repo_dir, cpu, mem_mb)
            if container:
                result = pool.exec_in_container(container, cmd_argv, timeout_sec)
                pool.release(container)
                return DockerResult(
                    ok=result["ok"],
                    exit_code=result["exit_code"],
                    stdout=result["stdout"],
                    stderr=result["stderr"],
                    timed_out=False,
                )
        except Exception as e:
            print(f"Warm pool failed, falling back to cold start: {e}")


    try:
        # Check the command against the security policy before running inside Docker.
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        is_allowed, reason = is_command_allowed(cmd_str)
        if not is_allowed:
            return DockerResult(
                ok=False,
                exit_code=1,
                stdout="",
                stderr=f"Command blocked by security policy: {reason}",
                timed_out=False,
            )

        # Build docker run command
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{sb.repo_dir}:/repo",
            "-w",
            "/repo",
        ]

        # Add cache volumes for faster dependency installs
        if use_cache:
            # Python pip cache and site-packages (CRITICAL: packages must persist between runs)
            docker_cmd.extend(
                [
                    "-v",
                    "pip-cache:/root/.cache/pip",
                    "-v",
                    "python-site-packages:/usr/local/lib/python3.11/site-packages",
                ]
            )
            # npm/yarn/pnpm cache
            docker_cmd.extend(
                [
                    "-v",
                    "npm-cache:/root/.npm",
                    "-v",
                    "yarn-cache:/usr/local/share/.cache/yarn",
                    "-v",
                    "pnpm-cache:/root/.local/share/pnpm/store",
                ]
            )

        # Resource limits
        docker_cmd.extend(
            [
                f"--cpus={cpu}",
                f"--memory={mem_mb}m",
                f"--pids-limit={pids}",
                # Limit container disk usage to prevent resource exhaustion
                "--storage-opt",
                "size=10G",
            ]
        )

        # Read-only mode with tmpfs
        if read_only:
            docker_cmd.append("--read-only")
            docker_cmd.append("--tmpfs=/tmp:rw,noexec,nosuid,size=512m")

        # Network control
        if not network:
            docker_cmd.append("--network=none")

        # Environment variables for optimization
        docker_cmd.extend(
            [
                "-e",
                "TZ=UTC",
                "-e",
                "PYTHONHASHSEED=0",
                "-e",
                "PIP_DISABLE_PIP_VERSION_CHECK=1",
                "-e",
                "PIP_NO_CACHE_DIR=0",  # Enable pip cache for speed
                "-e",
                "LC_ALL=C.UTF-8",
                "-e",
                "npm_config_cache=/root/.npm",
                "-e",
                "YARN_CACHE_FOLDER=/usr/local/share/.cache/yarn",
            ]
        )

        docker_cmd.extend([docker_image, *cmd_argv])
        
        # Debug: Print docker command
        print(f"DEBUG_DOCKER_CMD: {' '.join(docker_cmd)}", flush=True)

        # Remove secret API keys from the environment to prevent leakage to Docker
        # Env Allowlist: Only pass safe variables to Docker
        ALLOWED_ENV_VARS = {"PATH", "HOME", "LANG", "PYTHONPATH"}
        safe_env = {
            k: v for k, v in os.environ.items() if k in ALLOWED_ENV_VARS
        }
        # Run docker command with sanitized environment
        p = subprocess.run(
            docker_cmd,
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env=safe_env, check=False,
        )

        return DockerResult(
            ok=p.returncode == 0,
            exit_code=p.returncode,
            stdout=p.stdout,
            stderr=p.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return DockerResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout_sec}s",
            timed_out=True,
        )
    except FileNotFoundError:
        return DockerResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr="Docker not found. Please install Docker.",
            timed_out=False,
        )
    except Exception as e:
        return DockerResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr=f"Docker execution error: {e}",
            timed_out=False,
        )


def docker_install(
    sb: Sandbox,
    cmd: str,
    timeout_sec: int = 300,
    docker_image: str = "python:3.11-slim",
    cpu: float = 2.0,
    mem_mb: int = 4096,
    pids: int = 256,
    read_only: bool = False,
) -> DockerResult:
    """Run a dependency installation command with network enabled.

    Args:
        sb: The sandbox containing the repo.
        cmd: The install command to run.
        timeout_sec: Timeout for installation.
        docker_image: Docker image to use.
        cpu: CPU limit.
        mem_mb: Memory limit in MB.
        pids: Process ID limit.
        read_only: Whether to mount repo as read-only.

    Returns:
        DockerResult with installation status and output.
    """
    return docker_run(
        sb,
        cmd,
        timeout_sec=timeout_sec,
        network=True,
        docker_image=docker_image,
        cpu=cpu,
        mem_mb=mem_mb,
        pids=pids,
        read_only=read_only,
        use_cache=True,
    )


def docker_test(
    sb: Sandbox,
    cmd: str,
    timeout_sec: int = 120,
    docker_image: str = "python:3.11-slim",
    cpu: float = 2.0,
    mem_mb: int = 4096,
    pids: int = 256,
    read_only: bool = False,
) -> DockerResult:
    """Run a test command with network disabled.

    Args:
        sb: The sandbox containing the repo.
        cmd: The test command to run.
        timeout_sec: Timeout for tests.
        docker_image: Docker image to use.
        cpu: CPU limit.
        mem_mb: Memory limit in MB.
        pids: Process ID limit.
        read_only: Whether to mount repo as read-only.

    Returns:
        DockerResult with test status and output.
    """
    # Enable network for npx commands (need to download packages)
    network_enabled = cmd.startswith("npx ")
    return docker_run(
        sb,
        cmd,
        timeout_sec=timeout_sec,
        network=network_enabled,
        docker_image=docker_image,
        cpu=cpu,
        mem_mb=mem_mb,
        pids=pids,
        read_only=read_only,
        use_cache=True,
    )
