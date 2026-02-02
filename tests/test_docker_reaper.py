"""Tests for DockerReaper module."""

import subprocess
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from rfsn_controller.docker_reaper import ContainerInfo, DockerReaper


class TestDockerReaper:
    """Test suite for DockerReaper class."""

    def test_init(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        reaper = DockerReaper()
        assert reaper.label_filter == "rfsn-managed=true"
        assert reaper.max_age_hours == 24
        assert reaper.dry_run is False
        
        # Custom initialization
        reaper = DockerReaper(
            label_filter="custom-label=value",
            max_age_hours=48,
            dry_run=True
        )
        assert reaper.label_filter == "custom-label=value"
        assert reaper.max_age_hours == 48
        assert reaper.dry_run is True

    @patch('subprocess.run')
    def test_list_containers_success(self, mock_run):
        """Test successful container listing."""
        # Mock docker ps output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="abc123\t2026-01-29 10:30:45 -0800 PST\tUp 2 hours\ttest-container-1\n"
                   "def456\t2026-01-28 08:15:30 -0800 PST\tExited (0) 1 day ago\ttest-container-2\n",
            stderr=""
        )
        
        reaper = DockerReaper()
        containers = reaper.list_containers()
        
        assert len(containers) == 2
        assert containers[0].id == "abc123"
        assert containers[0].names == "test-container-1"
        assert containers[1].id == "def456"
        assert containers[1].status == "Exited (0) 1 day ago"

    @patch('subprocess.run')
    def test_list_containers_empty(self, mock_run):
        """Test container listing with no results."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )
        
        reaper = DockerReaper()
        containers = reaper.list_containers()
        
        assert len(containers) == 0

    @patch('subprocess.run')
    def test_list_containers_error(self, mock_run):
        """Test container listing with Docker error."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Cannot connect to Docker daemon"
        )
        
        reaper = DockerReaper()
        containers = reaper.list_containers()
        
        assert len(containers) == 0

    @patch('subprocess.run')
    def test_reap_old_containers(self, mock_run):
        """Test reaping containers older than threshold."""
        # Create a container older than threshold
        old_date = datetime.now() - timedelta(hours=48)
        date_str = old_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Mock docker ps to list containers
        list_result = Mock(
            returncode=0,
            stdout=f"old123\t{date_str} -0800 PST\tExited (0)\told-container\n",
            stderr=""
        )
        
        # Mock docker rm for removal
        rm_result = Mock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = [list_result, rm_result]
        
        reaper = DockerReaper(max_age_hours=24)
        reaped = reaper.reap()
        
        assert reaped == 1
        # Verify docker rm was called
        assert mock_run.call_count == 2

    @patch('subprocess.run')
    def test_reap_recent_containers_not_removed(self, mock_run):
        """Test that recent containers are not reaped."""
        # Create a recent container
        recent_date = datetime.now() - timedelta(hours=2)
        date_str = recent_date.strftime("%Y-%m-%d %H:%M:%S")
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=f"recent123\t{date_str} -0800 PST\tUp 2 hours\trecent-container\n",
            stderr=""
        )
        
        reaper = DockerReaper(max_age_hours=24)
        reaped = reaper.reap()
        
        assert reaped == 0
        # Only docker ps should be called, not docker rm
        assert mock_run.call_count == 1

    @patch('subprocess.run')
    def test_reap_dry_run(self, mock_run):
        """Test dry run mode does not remove containers."""
        old_date = datetime.now() - timedelta(hours=48)
        date_str = old_date.strftime("%Y-%m-%d %H:%M:%S")
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=f"old123\t{date_str} -0800 PST\tExited (0)\told-container\n",
            stderr=""
        )
        
        reaper = DockerReaper(max_age_hours=24, dry_run=True)
        reaped = reaper.reap()
        
        assert reaped == 0
        # Only docker ps should be called, not docker rm
        assert mock_run.call_count == 1

    @patch('subprocess.run')
    def test_reap_by_status_exited(self, mock_run):
        """Test reaping containers by status."""
        # Mock docker ps to list exited containers
        list_result = Mock(
            returncode=0,
            stdout="exit1\t2026-01-29 10:30:45 -0800 PST\tExited (0) 1 hour ago\ttest-1\n"
                   "exit2\t2026-01-29 09:15:30 -0800 PST\tExited (1) 2 hours ago\ttest-2\n",
            stderr=""
        )
        
        # Mock docker rm for removals
        rm_result = Mock(returncode=0, stdout="", stderr="")
        
        mock_run.side_effect = [list_result, rm_result, rm_result]
        
        reaper = DockerReaper()
        reaped = reaper.reap_by_status("exited")
        
        assert reaped == 2
        assert mock_run.call_count == 3  # 1 list + 2 removes

    @patch('subprocess.run')
    def test_reap_timeout_handling(self, mock_run):
        """Test handling of timeout during container removal."""
        old_date = datetime.now() - timedelta(hours=48)
        date_str = old_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Mock docker ps
        list_result = Mock(
            returncode=0,
            stdout=f"old123\t{date_str} -0800 PST\tExited (0)\told-container\n",
            stderr=""
        )
        
        # Mock docker rm with timeout
        def side_effect(*args, **kwargs):
            if "ps" in args[0]:
                return list_result
            else:
                raise subprocess.TimeoutExpired(cmd=args[0], timeout=60)
        
        mock_run.side_effect = side_effect
        
        reaper = DockerReaper(max_age_hours=24)
        reaped = reaper.reap()
        
        # Should handle timeout gracefully
        assert reaped == 0


class TestContainerInfo:
    """Test suite for ContainerInfo dataclass."""

    def test_container_info_creation(self):
        """Test ContainerInfo dataclass creation."""
        info = ContainerInfo(
            id="abc123",
            created_at=datetime.now(),
            status="Up 2 hours",
            names="test-container"
        )
        
        assert info.id == "abc123"
        assert isinstance(info.created_at, datetime)
        assert info.status == "Up 2 hours"
        assert info.names == "test-container"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
