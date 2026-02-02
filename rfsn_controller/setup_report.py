"""Setup report module for RFSN controller.
from __future__ import annotations

Creates structured reports about project setup and installation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SetupResult:
    """Result from a single setup command."""
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class SetupReport:
    """Report on project setup status."""
    pip_result: SetupResult | None = None
    node_result: SetupResult | None = None
    go_result: SetupResult | None = None
    rust_result: SetupResult | None = None
    java_result: SetupResult | None = None
    dotnet_result: SetupResult | None = None
    lockfile_path: str | None = None
    sysdeps_installed: list[str] = field(default_factory=list)
    sysdeps_failed: list[str] = field(default_factory=list)
    sysdeps_blocked: list[str] = field(default_factory=list)
    
    @property
    def has_lockfile(self) -> bool:
        """Check if a lockfile was found."""
        return self.lockfile_path is not None and len(self.lockfile_path) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        
        for lang in ["pip", "node", "go", "rust", "java", "dotnet"]:
            r = getattr(self, f"{lang}_result")
            if r is not None:
                # Handle both DockerResult (ok) and SetupResult (success)
                success = getattr(r, 'ok', getattr(r, 'success', True))
                output = getattr(r, 'stdout', getattr(r, 'output', ''))
                error = getattr(r, 'stderr', getattr(r, 'error', ''))
                result[f"{lang}_result"] = {
                    "success": success,
                    "output_length": len(output) if output else 0,
                    "error": error[:200] if error else "",
                }
        
        result["lockfile_path"] = self.lockfile_path
        result["sysdeps_installed"] = self.sysdeps_installed
        result["sysdeps_failed"] = self.sysdeps_failed
        result["sysdeps_blocked"] = self.sysdeps_blocked
        
        return result
    
    def should_bailout(self) -> bool:
        """Determine if setup failed badly enough to bail out."""
        # Check if any critical setup failed
        for lang in ["pip", "node", "go", "rust", "java", "dotnet"]:
            r = getattr(self, f"{lang}_result")
            if r is not None:
                success = getattr(r, 'ok', getattr(r, 'success', True))
                error = getattr(r, 'stderr', getattr(r, 'error', ''))
                if not success:
                    # Only bailout on critical failures
                    if error and "fatal" in error.lower():
                        return True
        
        # Check if too many sysdeps failed
        if len(self.sysdeps_failed) > 5:
            return True
        
        return False
    
    def get_bailout_message(self) -> str:
        """Get a message explaining why we're bailing out."""
        messages = []
        
        for lang in ["pip", "node", "go", "rust", "java", "dotnet"]:
            r = getattr(self, f"{lang}_result")
            if r is not None:
                success = getattr(r, 'ok', getattr(r, 'success', True))
                error = getattr(r, 'stderr', getattr(r, 'error', ''))
                if not success:
                    messages.append(f"{lang}: {error[:100] if error else 'failed'}")
        
        if self.sysdeps_failed:
            messages.append(f"Failed sysdeps: {', '.join(self.sysdeps_failed[:5])}")
        
        return "; ".join(messages) if messages else "Unknown setup failure"


def create_setup_report(
    pip_result: SetupResult | None = None,
    node_result: SetupResult | None = None,
    go_result: SetupResult | None = None,
    rust_result: SetupResult | None = None,
    java_result: SetupResult | None = None,
    dotnet_result: SetupResult | None = None,
    lockfile_path: str | None = None,
    sysdeps_installed: list[str] | None = None,
    sysdeps_failed: list[str] | None = None,
    sysdeps_blocked: list[str] | None = None,
) -> SetupReport:
    """Create a setup report from installation results.
    
    Args:
        pip_result: Result from pip installation
        node_result: Result from npm/yarn installation
        go_result: Result from go mod installation
        rust_result: Result from cargo installation
        java_result: Result from gradle/maven installation
        dotnet_result: Result from dotnet restore
        lockfile_path: Path to the lockfile if found
        sysdeps_installed: List of installed system dependencies
        sysdeps_failed: List of failed system dependency installations
        sysdeps_blocked: List of blocked system dependencies
        
    Returns:
        SetupReport instance with all results
    """
    return SetupReport(
        pip_result=pip_result,
        node_result=node_result,
        go_result=go_result,
        rust_result=rust_result,
        java_result=java_result,
        dotnet_result=dotnet_result,
        lockfile_path=lockfile_path,
        sysdeps_installed=sysdeps_installed or [],
        sysdeps_failed=sysdeps_failed or [],
        sysdeps_blocked=sysdeps_blocked or [],
    )
