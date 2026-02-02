"""
DockerReaper: Cleans up orphaned RFSN Docker containers.

This module provides automatic cleanup of Docker containers that were created
by RFSN but may have been left orphaned due to crashes or interruptions.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ContainerInfo:
    """Information about a Docker container."""
    
    id: str
    created_at: datetime
    status: str
    names: str


class DockerReaper:
    """
    Background process to reap orphaned RFSN containers.
    
    The reaper identifies containers with specific labels and removes them
    if they exceed a configured age threshold. This prevents resource leaks
    from long-running or abandoned containers.
    
    Example:
        >>> reaper = DockerReaper(label_filter="rfsn-managed=true", max_age_hours=24)
        >>> reaper.reap()
        2026-01-29 10:30:45 - Reaped 3 containers
    """

    def __init__(
        self, 
        label_filter: str = "rfsn-managed=true", 
        max_age_hours: int = 24,
        dry_run: bool = False
    ):
        """
        Initialize the Docker Reaper.
        
        Args:
            label_filter: Docker label filter to identify RFSN containers
            max_age_hours: Maximum age in hours before a container is reaped
            dry_run: If True, log what would be done without actually removing containers
        """
        self.label_filter = label_filter
        self.max_age_hours = max_age_hours
        self.dry_run = dry_run
        self.cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)

    def list_containers(self) -> list[ContainerInfo]:
        """
        List all containers matching the label filter.
        
        Returns:
            List of ContainerInfo objects for matching containers
        """
        try:
            cmd = [
                "docker", "ps", "-a",
                "--filter", f"label={self.label_filter}",
                "--format", "{{.ID}}\t{{.CreatedAt}}\t{{.Status}}\t{{.Names}}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if result.returncode != 0:
                logger.error(f"Failed to list containers: {result.stderr}")
                return []

            containers = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                    
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                    
                cid, created_str, status, names = parts
                
                # Parse creation date (format: "2026-01-29 10:30:45 -0800 PST")
                try:
                    # Extract just the date/time part
                    date_part = " ".join(created_str.split()[:2])
                    created_at = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(f"Could not parse date for container {cid}: {created_str}")
                    continue
                
                containers.append(ContainerInfo(
                    id=cid,
                    created_at=created_at,
                    status=status,
                    names=names
                ))
            
            return containers
            
        except subprocess.TimeoutExpired:
            logger.error("Docker command timed out")
            return []
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []

    def reap(self) -> int:
        """
        Find and remove containers matching the label that are too old.
        
        Returns:
            Number of containers reaped
        """
        containers = self.list_containers()
        reaped_count = 0
        
        for container in containers:
            # Make both datetimes comparable (handle naive vs aware)
            created_at = container.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if created_at < self.cutoff_time:
                if self.dry_run:
                    logger.info(
                        f"[DRY RUN] Would reap container {container.id} "
                        f"({container.names}), age: {datetime.now(UTC) - created_at}"
                    )
                else:
                    try:
                        logger.info(
                            f"Reaping container {container.id} ({container.names}), "
                            f"age: {datetime.now(UTC) - created_at}"
                        )
                        subprocess.run(
                            ["docker", "rm", "-f", container.id],
                            capture_output=True,
                            text=True,
                            timeout=60, check=False
                        )
                        reaped_count += 1
                    except subprocess.TimeoutExpired:
                        logger.error(f"Timeout removing container {container.id}")
                    except Exception as e:
                        logger.error(f"Error removing container {container.id}: {e}")
        
        if reaped_count > 0 or self.dry_run:
            logger.info(f"Reaped {reaped_count} containers")
        
        return reaped_count

    def reap_by_status(self, status_filter: str = "exited") -> int:
        """
        Reap containers by status (e.g., 'exited', 'dead').
        
        Args:
            status_filter: Container status to filter by
            
        Returns:
            Number of containers reaped
        """
        containers = self.list_containers()
        reaped_count = 0
        
        for container in containers:
            if status_filter.lower() in container.status.lower():
                if self.dry_run:
                    logger.info(
                        f"[DRY RUN] Would reap {status_filter} container {container.id}"
                    )
                else:
                    try:
                        subprocess.run(
                            ["docker", "rm", "-f", container.id],
                            capture_output=True,
                            timeout=60, check=False
                        )
                        reaped_count += 1
                    except Exception as e:
                        logger.error(f"Error removing container {container.id}: {e}")
        
        return reaped_count

if __name__ == "__main__":
    reaper = DockerReaper()
    reaper.reap()
