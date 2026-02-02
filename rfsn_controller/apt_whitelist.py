"""APT package whitelist for safe system dependency installation.
from __future__ import annotations

Organized in tiers from essential to specialized. Only packages in this
whitelist can be installed, and only through the safe SYSDEPS phase.
"""

from dataclasses import dataclass
from enum import Enum


class AptTier(Enum):
    """APT package tiers."""

    TIER_0 = "tier_0"  # Core build essentials
    TIER_1 = "tier_1"  # Crypto/compression headers
    TIER_2 = "tier_2"  # Database client headers
    TIER_3 = "tier_3"  # XML/parsing
    TIER_4 = "tier_4"  # Imaging/fonts
    TIER_5 = "tier_5"  # LDAP/Kerberos
    TIER_6 = "tier_6"  # Scientific Python
    TIER_7 = "tier_7"  # Networking


# Tier 0 - Core build essentials (almost always needed)
TIER_0_PACKAGES = [
    "build-essential",
    "pkg-config",
    "git",
    "ca-certificates",
    "python3",
    "python3-dev",
    "python3-venv",
    "python3-pip",
    "gcc",
    "g++",
    "make",
    "cmake",
    "ninja-build",
]

# Tier 1 - Common crypto/compression headers
TIER_1_PACKAGES = [
    "libssl-dev",
    "libffi-dev",
    "zlib1g-dev",
    "libbz2-dev",
    "liblzma-dev",
]

# Tier 2 - Database client headers
TIER_2_PACKAGES = [
    "libpq-dev",
    "default-libmysqlclient-dev",
    "libsqlite3-dev",
]

# Tier 3 - XML/parsing
TIER_3_PACKAGES = [
    "libxml2-dev",
    "libxslt1-dev",
    "libyaml-dev",
]

# Tier 4 - Imaging + fonts
TIER_4_PACKAGES = [
    "libjpeg-dev",
    "libpng-dev",
    "libfreetype6-dev",
    "libwebp-dev",
    "libtiff5-dev",
    "libopenjp2-7-dev",
]

# Tier 5 - LDAP/Kerberos (enterprise repos)
TIER_5_PACKAGES = [
    "libsasl2-dev",
    "libldap2-dev",
    "libkrb5-dev",
]

# Tier 6 - Scientific Python (NumPy/SciPy builds from source)
TIER_6_PACKAGES = [
    "gfortran",
    "libblas-dev",
    "liblapack-dev",
]

# Tier 7 - Networking libs
TIER_7_PACKAGES = [
    "libcurl4-openssl-dev",
]

# Packages that are NEVER allowed (services, daemons, dangerous tools)
FORBIDDEN_PACKAGES = {
    "postgresql",
    "redis-server",
    "mysql-server",
    "docker.io",
    "openssh-server",
    "nginx",
    "apache2",
    "snapd",
    "systemd",
    "iptables",
    "ufw",
    "sudo",
}


@dataclass
class AptWhitelist:
    """APT package whitelist configuration."""

    max_packages: int = 10
    max_tier: AptTier = AptTier.TIER_4
    allow_wildcards: bool = False
    custom_packages: list[str] | None = None

    def __post_init__(self):
        """Initialize the allowed packages set."""
        if self.custom_packages is None:
            self.custom_packages = []
        self._allowed: set[str] = set()
        self._build_allowed_set()

    def _build_allowed_set(self) -> None:
        """Build the set of allowed packages based on max_tier."""
        tiers_to_include = [
            AptTier.TIER_0,
            AptTier.TIER_1,
            AptTier.TIER_2,
            AptTier.TIER_3,
            AptTier.TIER_4,
            AptTier.TIER_5,
            AptTier.TIER_6,
            AptTier.TIER_7,
        ]

        for tier in tiers_to_include:
            if self._tier_value(tier) > self._tier_value(self.max_tier):
                break
            self._allowed.update(self._get_tier_packages(tier))

        # Add custom packages
        if self.custom_packages:
            self._allowed.update(self.custom_packages)

    def _tier_value(self, tier: AptTier) -> int:
        """Get numeric value for tier comparison."""
        tier_order = {
            AptTier.TIER_0: 0,
            AptTier.TIER_1: 1,
            AptTier.TIER_2: 2,
            AptTier.TIER_3: 3,
            AptTier.TIER_4: 4,
            AptTier.TIER_5: 5,
            AptTier.TIER_6: 6,
            AptTier.TIER_7: 7,
        }
        return tier_order.get(tier, 0)

    def _get_tier_packages(self, tier: AptTier) -> set[str]:
        """Get packages for a tier."""
        tier_map = {
            AptTier.TIER_0: set(TIER_0_PACKAGES),
            AptTier.TIER_1: set(TIER_1_PACKAGES),
            AptTier.TIER_2: set(TIER_2_PACKAGES),
            AptTier.TIER_3: set(TIER_3_PACKAGES),
            AptTier.TIER_4: set(TIER_4_PACKAGES),
            AptTier.TIER_5: set(TIER_5_PACKAGES),
            AptTier.TIER_6: set(TIER_6_PACKAGES),
            AptTier.TIER_7: set(TIER_7_PACKAGES),
        }
        return tier_map.get(tier, set())

    def is_allowed(self, package: str) -> bool:
        """Check if a package is allowed.

        Args:
            package: Package name to check.

        Returns:
            True if package is allowed.
        """
        # Check forbidden list first
        if package in FORBIDDEN_PACKAGES:
            return False

        # Check wildcards
        if "*" in package:
            if not self.allow_wildcards:
                return False
            # For wildcards, check if any package matches
            base = package.replace("*", "")
            return any(p.startswith(base) for p in self._allowed)

        return package in self._allowed

    def filter_allowed(self, packages: list[str]) -> tuple[list[str], list[str]]:
        """Filter packages into allowed and blocked.

        Args:
            packages: List of package names.

        Returns:
            (allowed_packages, blocked_packages) tuple.
        """
        allowed = []
        blocked = []

        for pkg in packages:
            if self.is_allowed(pkg):
                allowed.append(pkg)
            else:
                blocked.append(pkg)

        return allowed, blocked

    def check_within_limits(self, packages: list[str]) -> bool:
        """Check if number of packages is within limits.

        Args:
            packages: List of package names.

        Returns:
            True if within limits.
        """
        return len(packages) <= self.max_packages


# Default whitelist (Tier 0-4, max 10 packages)
DEFAULT_WHITELIST = AptWhitelist(
    max_packages=10,
    max_tier=AptTier.TIER_4,
    allow_wildcards=False,
)


# Conservative whitelist (Tier 0-2, max 5 packages)
CONSERVATIVE_WHITELIST = AptWhitelist(
    max_packages=5,
    max_tier=AptTier.TIER_2,
    allow_wildcards=False,
)


# Permissive whitelist (Tier 0-7, max 20 packages)
PERMISSIVE_WHITELIST = AptWhitelist(
    max_packages=20,
    max_tier=AptTier.TIER_7,
    allow_wildcards=False,
)


def get_starter_packages() -> list[str]:
    """Get starter package set that covers most Python repos.

    Returns:
        List of package names.
    """
    return [
        "build-essential",
        "pkg-config",
        "git",
        "ca-certificates",
        "python3",
        "python3-dev",
        "python3-venv",
        "python3-pip",
        "cmake",
        "ninja-build",
        "libssl-dev",
        "libffi-dev",
        "zlib1g-dev",
        "libbz2-dev",
        "liblzma-dev",
        "libpq-dev",
        "default-libmysqlclient-dev",
        "libxml2-dev",
        "libxslt1-dev",
        "libyaml-dev",
        "libjpeg-dev",
        "libpng-dev",
        "libfreetype6-dev",
        "libwebp-dev",
        "libtiff5-dev",
        "libopenjp2-7-dev",
        "libsasl2-dev",
        "libldap2-dev",
        "libkrb5-dev",
        "libcurl4-openssl-dev",
    ]
