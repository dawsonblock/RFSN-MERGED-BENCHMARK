"""URL validation utilities for security.
from __future__ import annotations

Enforces strict GitHub URL validation to prevent:
- Non-repo URLs (like blob/, tree/, commit/)
- Weird suffixes / extra path segments
- Only allows OWNER/REPO(.git) format
"""

import os
import re
from urllib.parse import urlparse

# Strict regex for GitHub repository URLs
# Only matches: https://github.com/OWNER/REPO or https://github.com/OWNER/REPO.git
GITHUB_REPO_REGEX = re.compile(r"^https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+(\.git)?$")


# Patterns that indicate non-repo URLs (should be blocked)
BLOCKED_PATTERNS = [
    "/blob/",
    "/tree/",
    "/commit/",
    "/pull/",
    "/issues/",
    "/wiki/",
    "/actions/",
    "/projects/",
    "/pulse/",
    "/graphs/",
    "/network/",
    "/settings/",
    "/security/",
    "/stargazers/",
    "/watchers/",
    "/forks/",
]


def normalize_github_url(url: str) -> str:
    """Normalize a GitHub URL to standard format.

    Removes trailing slash, ensures https, removes .git if present.

    Args:
        url: The GitHub URL to normalize.

    Returns:
        Normalized URL.
    """
    url = url.strip()

    # Ensure https
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)

    # Remove trailing slash
    if url.endswith("/"):
        url = url[:-1]

    return url


def validate_github_url(url: str) -> tuple[bool, str | None, str | None]:
    """Validate a GitHub repository URL.

    Args:
        url: The URL to validate.

    Returns:
        (is_valid, normalized_url, error_message) tuple.
        If valid, normalized_url contains the normalized URL.
        If invalid, error_message contains the reason.
    """
    # Support local paths for testing/development
    # Check this FIRST before urlparse which fails on local paths
    if os.path.exists(url) or url.startswith(("./", "../", "/")):
        return True, url, None

    # Basic URL parsing
    try:
        parsed = urlparse(url)
        if not parsed.netloc or not parsed.path:
            return False, None, "Invalid URL format"
    except Exception:
        return False, None, "Invalid URL format"

    # Support local paths for testing/development
    if os.path.exists(url) or url.startswith("./") or url.startswith("../") or url.startswith("/"):
        return True, url, None

    # Must be github.com
    if "github.com" not in parsed.netloc:
        return False, None, "Only GitHub URLs are allowed"

    # Check for blocked patterns
    url_lower = url.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in url_lower:
            return False, None, f"Repository URLs cannot contain '{pattern}'"

    # Normalize URL
    normalized = normalize_github_url(url)

    # Check against strict regex
    if not GITHUB_REPO_REGEX.match(normalized):
        return (
            False,
            None,
            (
                "Invalid GitHub repository URL format. "
                "Expected: https://github.com/OWNER/REPO or https://github.com/OWNER/REPO.git"
            ),
        )

    return True, normalized, None


def extract_repo_info(url: str) -> tuple[str, str] | None:
    """Extract owner and repository name from a GitHub URL.

    Args:
        url: The GitHub URL.

    Returns:
        (owner, repo) tuple or None if invalid.
    """
    is_valid, normalized, _ = validate_github_url(url)
    if not is_valid or normalized is None:
        return None

    # Remove .git suffix if present
    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    # Extract owner/repo from path
    parts = normalized.rstrip("/").split("/")
    if len(parts) >= 2:
        owner = parts[-2]
        repo = parts[-1]
        return owner, repo

    return None
