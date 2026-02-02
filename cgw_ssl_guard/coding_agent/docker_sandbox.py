"""Docker Sandbox Integration for CGW Coding Agent.

This module provides a Docker-based sandbox that implements the
SandboxProtocol required by the BlockingExecutor. It enables
safe, isolated code execution for the serial decision agent.

Features:
- Disposable container per session
- Git repository cloning
- Test execution with timeout
- Patch application
- File read/write operations
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DockerSandboxConfig:
    """Configuration for Docker sandbox."""
    
    # Docker image
    image: str = "python:3.11-slim"
    
    # Container resource limits
    memory_limit: str = "2g"
    cpu_limit: str = "2"
    
    # Timeouts (seconds)
    container_timeout: int = 600
    command_timeout: int = 300
    clone_timeout: int = 120
    
    # Working directories
    container_workdir: str = "/workspace"
    host_temp_dir: Optional[str] = None
    
    # Network access
    network_disabled: bool = False
    
    # Cleanup
    auto_cleanup: bool = True


class DockerSandbox:
    """Docker-based sandbox for code execution.
    
    Implements SandboxProtocol for use with BlockingExecutor.
    Creates a disposable container that can run arbitrary commands,
    apply patches, and execute tests in isolation.
    
    Usage:
        sandbox = DockerSandbox(config)
        sandbox.start(repo_url="https://github.com/user/repo")
        result = sandbox.run("pytest -q")
        sandbox.stop()
    """
    
    def __init__(self, config: Optional[DockerSandboxConfig] = None):
        self.config = config or DockerSandboxConfig()
        self._container_id: Optional[str] = None
        self._temp_dir: Optional[Path] = None
        self._is_running = False
    
    def start(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        checkout: Optional[str] = None,
    ) -> str:
        """Start the sandbox container.
        
        Args:
            repo_url: Optional Git repository to clone.
            branch: Branch to checkout.
            checkout: Specific commit/tag to checkout.
            
        Returns:
            Container ID.
            
        Raises:
            RuntimeError: If container fails to start.
        """
        if self._is_running:
            raise RuntimeError("Sandbox already running")
        
        # Create temp directory for volume mount
        self._temp_dir = Path(tempfile.mkdtemp(
            prefix="cgw_sandbox_",
            dir=self.config.host_temp_dir,
        ))
        
        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--memory", self.config.memory_limit,
            "--cpus", self.config.cpu_limit,
            "-v", f"{self._temp_dir}:{self.config.container_workdir}",
            "-w", self.config.container_workdir,
        ]
        
        if self.config.network_disabled:
            cmd.append("--network=none")
        
        # Use tail -f /dev/null to keep container running
        cmd.extend([self.config.image, "tail", "-f", "/dev/null"])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Docker run failed: {result.stderr}")
            
            self._container_id = result.stdout.strip()[:12]
            self._is_running = True
            logger.info(f"Started sandbox container: {self._container_id}")
            
            # Clone repository if provided
            if repo_url:
                self._clone_repo(repo_url, branch, checkout)
            
            return self._container_id
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker run timed out")
    
    def _clone_repo(
        self,
        repo_url: str,
        branch: str = "main",
        checkout: Optional[str] = None,
    ) -> None:
        """Clone a git repository into the container."""
        # Clone command
        clone_cmd = f"git clone --depth 1 --branch {branch} {repo_url} ."
        result = self.run(clone_cmd, timeout=self.config.clone_timeout)
        
        if result.returncode != 0:
            # Try without branch specification
            clone_cmd = f"git clone --depth 1 {repo_url} ."
            result = self.run(clone_cmd, timeout=self.config.clone_timeout)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {result.stderr}")
        
        # Checkout specific commit if provided
        if checkout:
            result = self.run(f"git fetch origin {checkout} && git checkout {checkout}")
            if result.returncode != 0:
                logger.warning(f"Failed to checkout {checkout}")
    
    def run(
        self,
        cmd: str,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in the sandbox.
        
        This is the main interface for SandboxProtocol.
        
        Args:
            cmd: Shell command to run.
            timeout: Command timeout in seconds.
            
        Returns:
            CompletedProcess with stdout, stderr, and returncode.
        """
        if not self._is_running:
            raise RuntimeError("Sandbox not running")
        
        timeout = timeout or self.config.command_timeout
        
        docker_cmd = [
            "docker", "exec",
            "-w", self.config.container_workdir,
            self._container_id,
            "/bin/bash", "-c", cmd,
        ]
        
        try:
            return subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out after {timeout}s: {cmd[:100]}")
            return subprocess.CompletedProcess(
                args=docker_cmd,
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
            )
    
    def read_file(self, path: str) -> str:
        """Read a file from the sandbox.
        
        Args:
            path: Path relative to workdir, or absolute in container.
            
        Returns:
            File contents.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not self._is_running:
            raise RuntimeError("Sandbox not running")
        
        result = self.run(f"cat {path}")
        if result.returncode != 0:
            raise FileNotFoundError(f"File not found or unreadable: {path}")
        
        return result.stdout
    
    def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox.
        
        Args:
            path: Path relative to workdir, or absolute in container.
            content: File contents to write.
        """
        if not self._is_running:
            raise RuntimeError("Sandbox not running")
        
        # Use heredoc for safe content transfer
        escaped = content.replace("'", "'\"'\"'")
        cmd = f"cat > {path} << 'CGWEOF'\n{escaped}\nCGWEOF"
        
        result = self.run(cmd)
        if result.returncode != 0:
            raise IOError(f"Failed to write file: {result.stderr}")
    
    def apply_diff(self, diff: str) -> bool:
        """Apply a unified diff to the codebase.
        
        Args:
            diff: Unified diff content.
            
        Returns:
            True if patch applied successfully.
        """
        if not self._is_running:
            raise RuntimeError("Sandbox not running")
        
        # Write diff to temp file
        diff_path = "/tmp/cgw_patch.diff"
        self.write_file(diff_path, diff)
        
        # Apply with git apply (more lenient than patch)
        result = self.run(f"git apply --allow-empty -v {diff_path}")
        
        if result.returncode != 0:
            # Try with patch command as fallback
            result = self.run(f"patch -p1 < {diff_path}")
        
        return result.returncode == 0
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """List files matching pattern.
        
        Args:
            pattern: Glob pattern.
            
        Returns:
            List of file paths.
        """
        result = self.run(f"find . -name '{pattern}' -type f")
        if result.returncode != 0:
            return []
        return [f.strip() for f in result.stdout.split("\n") if f.strip()]
    
    def get_git_diff(self) -> str:
        """Get current git diff of changes.
        
        Returns:
            Unified diff of uncommitted changes.
        """
        result = self.run("git diff")
        return result.stdout if result.returncode == 0 else ""
    
    def stop(self) -> None:
        """Stop and remove the sandbox container."""
        if not self._is_running:
            return
        
        try:
            # Stop container
            subprocess.run(
                ["docker", "stop", "-t", "1", self._container_id],
                capture_output=True,
                timeout=10,
            )
            
            # Remove container
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                timeout=10,
            )
            
            logger.info(f"Stopped sandbox container: {self._container_id}")
            
        except Exception as e:
            logger.warning(f"Error stopping container: {e}")
        finally:
            self._is_running = False
            self._container_id = None
            
            # Cleanup temp directory
            if self.config.auto_cleanup and self._temp_dir:
                import shutil
                try:
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir: {e}")
    
    def __enter__(self) -> "DockerSandbox":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
    
    @property
    def is_running(self) -> bool:
        """Check if sandbox is running."""
        return self._is_running
    
    @property
    def container_id(self) -> Optional[str]:
        """Get container ID."""
        return self._container_id


class SandboxManager:
    """Manages multiple sandbox instances.
    
    Provides a pool of sandboxes for concurrent sessions
    and handles lifecycle management.
    """
    
    def __init__(
        self,
        max_sandboxes: int = 5,
        config: Optional[DockerSandboxConfig] = None,
    ):
        self.max_sandboxes = max_sandboxes
        self.config = config or DockerSandboxConfig()
        self._sandboxes: Dict[str, DockerSandbox] = {}
    
    def create(self, session_id: str, repo_url: Optional[str] = None) -> DockerSandbox:
        """Create a new sandbox for a session.
        
        Args:
            session_id: Unique session identifier.
            repo_url: Optional repository to clone.
            
        Returns:
            New DockerSandbox instance.
        """
        if len(self._sandboxes) >= self.max_sandboxes:
            # Cleanup oldest sandbox
            oldest = next(iter(self._sandboxes.keys()))
            self.destroy(oldest)
        
        sandbox = DockerSandbox(self.config)
        sandbox.start(repo_url=repo_url)
        self._sandboxes[session_id] = sandbox
        
        return sandbox
    
    def get(self, session_id: str) -> Optional[DockerSandbox]:
        """Get an existing sandbox by session ID."""
        return self._sandboxes.get(session_id)
    
    def destroy(self, session_id: str) -> None:
        """Destroy a sandbox by session ID."""
        sandbox = self._sandboxes.pop(session_id, None)
        if sandbox:
            sandbox.stop()
    
    def destroy_all(self) -> None:
        """Destroy all sandboxes."""
        for session_id in list(self._sandboxes.keys()):
            self.destroy(session_id)
    
    def list_sessions(self) -> List[str]:
        """List active session IDs."""
        return list(self._sandboxes.keys())


def check_docker_available() -> bool:
    """Check if Docker is available on the system.
    
    Returns:
        True if docker is installed and responsive.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_executor_sandbox(
    repo_url: Optional[str] = None,
    config: Optional[DockerSandboxConfig] = None,
) -> DockerSandbox:
    """Convenience function to create a sandbox for the BlockingExecutor.
    
    Usage:
        from cgw_ssl_guard.coding_agent.docker_sandbox import create_executor_sandbox
        from cgw_ssl_guard.coding_agent.executor import BlockingExecutor
        
        sandbox = create_executor_sandbox(repo_url="https://github.com/user/repo")
        executor = BlockingExecutor(sandbox=sandbox)
        
        # ... run agent ...
        
        sandbox.stop()
    """
    if not check_docker_available():
        logger.warning("Docker not available, sandbox features will be limited")
        return None
    
    sandbox = DockerSandbox(config)
    if repo_url:
        sandbox.start(repo_url=repo_url)
    
    return sandbox
