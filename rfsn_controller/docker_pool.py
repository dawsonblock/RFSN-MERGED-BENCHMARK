"""Docker Warm Container Pool for fast test execution.

Provides container reuse to eliminate cold-start overhead (2-5s per test run).
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class WarmContainer:
    """A pre-warmed Docker container."""
    
    container_id: str
    image: str
    created_at: float
    in_use: bool = False
    last_used_at: float | None = None


@dataclass
class WarmContainerPool:
    """Pool of warm Docker containers for fast execution.
    
    Instead of starting a new container for each test run, we keep a pool
    of warm containers that can be reused. This eliminates the 2-5s Docker
    startup overhead.
    
    Usage:
        pool = WarmContainerPool()
        container = pool.get_or_create("python:3.11-slim")
        result = pool.exec_in_container(container, ["pytest", "-x"])
        pool.release(container)  # Return to pool for reuse
    """
    
    pool_size: int = 3
    ttl_seconds: int = 300  # Keep containers warm for 5 minutes
    
    _containers: dict[str, list[WarmContainer]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def get_or_create(
        self,
        image: str,
        repo_dir: str,
        cpu: float = 2.0,
        mem_mb: int = 4096,
    ) -> WarmContainer | None:
        """Get an existing warm container or create a new one.
        
        Args:
            image: Docker image name.
            repo_dir: Path to repo directory to mount.
            cpu: CPU limit.
            mem_mb: Memory limit in MB.
            
        Returns:
            A warm container ready for execution, or None on failure.
        """
        with self._lock:
            # Check for existing available container
            if image in self._containers:
                for container in self._containers[image]:
                    if not container.in_use:
                        # Check if container is still alive
                        if self._is_alive(container.container_id):
                            container.in_use = True
                            return container
                        else:
                            # Remove dead container
                            self._containers[image].remove(container)
            
            # Create a new one
            container = self._create_container(image, repo_dir, cpu, mem_mb)
            if container:
                if image not in self._containers:
                    self._containers[image] = []
                self._containers[image].append(container)
            return container
    
    def release(self, container: WarmContainer) -> None:
        """Release a container back to the pool.
        
        Args:
            container: The container to release.
        """
        with self._lock:
            container.in_use = False
            container.last_used_at = time.time()
    
    def exec_in_container(
        self,
        container: WarmContainer,
        cmd: list[str],
        timeout_sec: int = 120,
    ) -> dict:
        """Execute a command in a warm container.
        
        Args:
            container: The warm container.
            cmd: Command to execute.
            timeout_sec: Timeout in seconds.
            
        Returns:
            Dict with ok, exit_code, stdout, stderr.
        """
        try:
            docker_cmd = [
                "docker", "exec",
                container.container_id,
                *cmd
            ]
            
            p = subprocess.run(
                docker_cmd,
                shell=False,
                text=True,
                capture_output=True,
                timeout=timeout_sec, check=False,
            )
            
            return {
                "ok": p.returncode == 0,
                "exit_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_sec}s",
            }
        except Exception as e:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired containers from the pool.
        
        Returns:
            Number of containers removed.
        """
        now = time.time()
        removed = 0
        
        with self._lock:
            for image, containers in list(self._containers.items()):
                for container in containers[:]:
                    if not container.in_use:
                        age = now - container.created_at
                        if age > self.ttl_seconds:
                            self._stop_container(container.container_id)
                            containers.remove(container)
                            removed += 1
        
        return removed
    
    def cleanup_all(self) -> None:
        """Stop and remove all containers in the pool."""
        with self._lock:
            for image, containers in self._containers.items():
                for container in containers:
                    self._stop_container(container.container_id)
            self._containers.clear()
    
    def _create_container(
        self,
        image: str,
        repo_dir: str,
        cpu: float,
        mem_mb: int,
    ) -> WarmContainer | None:
        """Create a new warm container."""
        try:
            # Create container in detached mode with sleep infinity
            docker_cmd = [
                "docker", "run",
                "-d",  # Detached
                "-v", f"{repo_dir}:/repo",
                "-w", "/repo",
                f"--cpus={cpu}",
                f"--memory={mem_mb}m",
                "-v", "pip-cache:/root/.cache/pip",
                "-v", "python-site-packages:/usr/local/lib/python3.11/site-packages",
                image,
                "sleep", "infinity"  # Keep container alive
            ]
            
            p = subprocess.run(
                docker_cmd,
                shell=False,
                text=True,
                capture_output=True,
                timeout=60, check=False,
            )
            
            if p.returncode == 0:
                container_id = p.stdout.strip()[:12]  # Short ID
                return WarmContainer(
                    container_id=container_id,
                    image=image,
                    created_at=time.time(),
                    in_use=True,
                )
            return None
        except Exception:
            return None
    
    def _is_alive(self, container_id: str) -> bool:
        """Check if a container is still running."""
        try:
            p = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                shell=False,
                text=True,
                capture_output=True,
                timeout=10, check=False,
            )
            return p.returncode == 0 and p.stdout.strip() == "true"
        except Exception:
            return False
    
    def _stop_container(self, container_id: str) -> None:
        """Stop and remove a container."""
        try:
            subprocess.run(
                ["docker", "stop", container_id],
                shell=False,
                capture_output=True,
                timeout=10, check=False,
            )
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                shell=False,
                capture_output=True,
                timeout=10, check=False,
            )
        except Exception:
            pass


# Global pool instance
_pool: WarmContainerPool | None = None
_pool_lock = threading.Lock()


def get_warm_pool() -> WarmContainerPool:
    """Get the global warm container pool instance."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = WarmContainerPool()
        return _pool
