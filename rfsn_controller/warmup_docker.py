"""Docker Warm Container Pre-initialization Script.

Run this script before benchmark execution to pre-warm Docker containers,
eliminating the 2-5 second cold-start overhead per task.

Usage:
    python -m rfsn_controller.warmup_docker
    python -m rfsn_controller.warmup_docker --image python:3.11-slim --count 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .docker_pool import get_warm_pool


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            "docker info",
            shell=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def pull_image_if_needed(image: str) -> bool:
    """Pull Docker image if not present locally."""
    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.stdout.strip():
        print(f"✓ Image {image} already present")
        return True
    
    print(f"Pulling {image}...")
    result = subprocess.run(
        ["docker", "pull", image],
        timeout=300,
        check=False,
    )
    return result.returncode == 0


def warmup_containers(
    image: str = "python:3.11-slim",
    count: int = 3,
    repo_dir: str = "/tmp/warmup_repo",
) -> int:
    """Pre-warm Docker containers for faster execution.
    
    Args:
        image: Docker image to warm up.
        count: Number of containers to pre-create.
        repo_dir: Dummy repo directory for container mount.
        
    Returns:
        Number of containers successfully warmed.
    """
    import os
    os.makedirs(repo_dir, exist_ok=True)
    
    pool = get_warm_pool()
    pool.pool_size = max(pool.pool_size, count)
    pool.ttl_seconds = 600  # Keep warm for 10 minutes
    
    warmed = 0
    for i in range(count):
        print(f"Warming container {i+1}/{count}...", end=" ")
        container = pool.get_or_create(image, repo_dir, cpu=2.0, mem_mb=4096)
        if container:
            pool.release(container)  # Release back to pool immediately
            print(f"✓ {container.container_id}")
            warmed += 1
        else:
            print("✗ Failed")
    
    return warmed


def show_pool_status() -> None:
    """Display current warm pool status."""
    result = subprocess.run(
        ["docker", "ps", "--filter", "ancestor=python:3.11-slim", "--format", "{{.ID}}\t{{.Status}}\t{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.stdout.strip():
        print("\nActive Python containers:")
        print("ID\t\tStatus\t\t\tName")
        print("-" * 50)
        print(result.stdout)
    else:
        print("\nNo active Python containers")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-warm Docker containers for RFSN benchmark execution"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="python:3.11-slim",
        help="Docker image to warm up (default: python:3.11-slim)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of containers to pre-create (default: 3)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pool status and exit",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up all warm containers and exit",
    )
    args = parser.parse_args()
    
    # Check Docker availability
    if not check_docker_available():
        print("ERROR: Docker is not available or not running")
        print("Please start Docker Desktop or ensure docker daemon is running")
        return 1
    
    if args.status:
        show_pool_status()
        return 0
    
    if args.cleanup:
        pool = get_warm_pool()
        pool.cleanup_all()
        print("✓ All warm containers cleaned up")
        return 0
    
    # Pull image if needed
    if not pull_image_if_needed(args.image):
        print(f"ERROR: Failed to pull image {args.image}")
        return 1
    
    # Warm up containers
    print(f"\nWarming up {args.count} containers with {args.image}...")
    start = time.time()
    warmed = warmup_containers(args.image, args.count)
    elapsed = time.time() - start
    
    print(f"\n✓ Warmed {warmed}/{args.count} containers in {elapsed:.1f}s")
    print("Containers will stay warm for 10 minutes or until cleanup")
    
    show_pool_status()
    return 0 if warmed == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
