"""System dependency installation phase.
from __future__ import annotations

Safely installs APT packages from a curated whitelist during the SYSDEPS phase.
"""

import re
import subprocess
from dataclasses import dataclass

from rfsn_controller.apt_whitelist import DEFAULT_WHITELIST, AptWhitelist


@dataclass
class SysdepsResult:
    """Result of system dependency installation."""

    success: bool
    installed_packages: list[str]
    blocked_packages: list[str]
    error_message: str | None


class SysdepsInstaller:
    """Installs system dependencies safely using APT whitelist."""

    def __init__(
        self,
        whitelist: AptWhitelist | None = None,
        dry_run: bool = False,
    ):
        """Initialize the installer.

        Args:
            whitelist: APT whitelist to use (default: DEFAULT_WHITELIST).
            dry_run: If True, don't actually install packages.
        """
        self.whitelist = whitelist or DEFAULT_WHITELIST
        self.dry_run = dry_run

    def parse_error_for_packages(self, error_output: str) -> list[str]:
        """Parse error output for missing package hints.

        Args:
            error_output: Error output from failed install.

        Returns:
            List of suggested package names.
        """
        packages = []

        # Common patterns for missing packages
        patterns = [
            r"lib([a-z0-9]+)-dev",
            r"package '([a-z0-9\-]+)' not found",
            r"unable to locate package ([a-z0-9\-]+)",
            r"E: Unable to locate package ([a-z0-9\-]+)",
            r"fatal error: ([a-z0-9_\/]+\.h): No such file",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, error_output, re.IGNORECASE)
            for match in matches:
                # Reconstruct full package name if needed
                if "-dev" not in match and "fatal error" in pattern:
                    # Extract library name from path
                    lib_name = match.split("/")[-1].split(".")[0]
                    pkg = f"lib{lib_name}-dev"
                else:
                    pkg = match
                packages.append(pkg)

        return list(set(packages))

    def install(
        self,
        packages: list[str],
        hints: list[str] | None = None,
    ) -> SysdepsResult:
        """Install packages safely with whitelist validation.

        Args:
            packages: List of package names to install.
            hints: Optional hints from error parsing.

        Returns:
            SysdepsResult with installation status.
        """
        # Combine explicit packages and hints
        all_packages = list(set(packages + (hints or [])))

        # Filter against whitelist
        allowed, blocked = self.whitelist.filter_allowed(all_packages)

        # Check limits
        if not self.whitelist.check_within_limits(allowed):
            return SysdepsResult(
                success=False,
                installed_packages=[],
                blocked_packages=blocked + allowed,
                error_message=(
                    f"Too many packages requested: {len(allowed)} > {self.whitelist.max_packages}"
                ),
            )

        # Dry run - just report what would happen
        if self.dry_run:
            return SysdepsResult(
                success=True,
                installed_packages=allowed,
                blocked_packages=blocked,
                error_message=None,
            )

        # Actually install packages
        if not allowed:
            return SysdepsResult(
                success=True,
                installed_packages=[],
                blocked_packages=blocked,
                error_message=None,
            )

        return self._run_apt_install(allowed, blocked)

    def _run_apt_install(
        self,
        packages: list[str],
        blocked: list[str],
    ) -> SysdepsResult:
        """Run apt-get install command.

        Args:
            packages: List of allowed packages to install.
            blocked: List of blocked packages.

        Returns:
            SysdepsResult with installation status.
        """
        try:
            # Update package list first
            update_cmd = ["apt-get", "update", "-qq"]
            result = subprocess.run(
                update_cmd,
                capture_output=True,
                text=True,
                timeout=300, check=False,
            )

            if result.returncode != 0:
                return SysdepsResult(
                    success=False,
                    installed_packages=[],
                    blocked_packages=blocked + packages,
                    error_message=f"apt-get update failed: {result.stderr}",
                )

            # Validate package names strictly to prevent injection
            PKG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-+.]+$")
            validated_packages = []
            for pkg in packages:
                if PKG_NAME_RE.match(pkg):
                    validated_packages.append(pkg)
                else:
                    # Log security warning (in a real system this would be logged to security audit)
                    print(f"SECURITY WARNING: Rejected invalid package name: {pkg}")
            
            if not validated_packages:
                 return SysdepsResult(
                    success=True, # Nothing valid to install is technically a success (no-op)
                    installed_packages=[],
                    blocked_packages=blocked + [p for p in packages if p not in validated_packages],
                    error_message=None,
                )

            # Install packages
            install_cmd = [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "-qq",
                "--",  # End of options, strictly positional arguments follow
            ] + validated_packages

            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=600, check=False,
            )

            if result.returncode != 0:
                # Try to parse error for missing packages
                self.parse_error_for_packages(result.stderr)

                return SysdepsResult(
                    success=False,
                    installed_packages=[],
                    blocked_packages=blocked + packages,
                    error_message=(f"apt-get install failed: {result.stderr.strip()}"),
                )

            return SysdepsResult(
                success=True,
                installed_packages=packages,
                blocked_packages=blocked,
                error_message=None,
            )

        except subprocess.TimeoutExpired:
            return SysdepsResult(
                success=False,
                installed_packages=[],
                blocked_packages=blocked + packages,
                error_message="apt-get install timed out",
            )
        except Exception as e:
            return SysdepsResult(
                success=False,
                installed_packages=[],
                blocked_packages=blocked + packages,
                error_message=f"Unexpected error: {e!s}",
            )

    def install_starter_set(self) -> SysdepsResult:
        """Install the starter package set for general use.

        Returns:
            SysdepsResult with installation status.
        """
        from rfsn_controller.apt_whitelist import get_starter_packages

        starter_packages = get_starter_packages()
        return self.install(starter_packages)

    def get_install_report(self) -> str:
        """Get a report of the whitelist configuration.

        Returns:
            String report of whitelist settings.
        """
        return (
            f"APT Whitelist Configuration:\n"
            f"  Max packages: {self.whitelist.max_packages}\n"
            f"  Max tier: {self.whitelist.max_tier.value}\n"
            f"  Allow wildcards: {self.whitelist.allow_wildcards}\n"
            f"  Dry run: {self.dry_run}\n"
        )
