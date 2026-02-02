"""Tool request deduplication and quota management.
from __future__ import annotations

Prevents model from spamming the same tool requests repeatedly
and enforces quotas to control token usage and prevent stalling.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolRequestConfig:
    """Configuration for tool request quotas."""

    max_requests_per_response: int = 6
    max_total_requests_per_run: int = 20
    enable_deduplication: bool = True


@dataclass
class ToolRequest:
    """A tool request with signature."""

    tool: str
    args: dict[str, Any]

    def signature(self) -> str:
        """Generate a unique signature for this request."""
        # Use a stable, recursive JSON representation for the entire payload
        # This ensures that {'a':1, 'b':2} and {'b':2, 'a':1} hash identically
        stable_json = json.dumps(
            {"tool": self.tool, "args": self.args}, 
            sort_keys=True, 
            default=str  # Fallback for non-serializable types
        )
        return hashlib.sha256(stable_json.encode()).hexdigest()


class ToolRequestManager:
    """Manages tool request deduplication and quotas."""

    def __init__(self, config: ToolRequestConfig | None = None):
        self.config = config or ToolRequestConfig()
        self.seen_signatures: set[str] = set()
        self.total_requests_this_run: int = 0
        self.request_counts: dict[str, int] = {}

    def should_allow_request(
        self,
        tool: str,
        args: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Check if a tool request should be allowed.

        Args:
            tool: The tool name.
            args: The tool arguments.

        Returns:
            (is_allowed, reason) tuple.
        """
        # Check total quota
        if self.total_requests_this_run >= self.config.max_total_requests_per_run:
            return False, (
                f"Total tool request quota exceeded: "
                f"{self.total_requests_this_run} >= {self.config.max_total_requests_per_run}"
            )

        # Create request and check deduplication
        request = ToolRequest(tool=tool, args=args)
        signature = request.signature()

        if self.config.enable_deduplication and signature in self.seen_signatures:
            return False, f"Duplicate request blocked: {tool}"

        return True, None

    def register_request(self, tool: str, args: dict[str, Any]) -> None:
        """Register a tool request as seen.

        Args:
            tool: The tool name.
            args: The tool arguments.
        """
        request = ToolRequest(tool=tool, args=args)
        signature = request.signature()

        if self.config.enable_deduplication:
            self.seen_signatures.add(signature)

        self.total_requests_this_run += 1
        self.request_counts[tool] = self.request_counts.get(tool, 0) + 1

    def filter_requests(
        self,
        requests: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Filter a list of tool requests against quotas and deduplication.

        Args:
            requests: List of tool request dictionaries.

        Returns:
            (allowed_requests, blocked_reasons) tuple.
        """
        allowed = []
        blocked = []

        # Check per-response limit
        if len(requests) > self.config.max_requests_per_response:
            blocked.append(
                f"Too many requests in response: {len(requests)} > "
                f"{self.config.max_requests_per_response}"
            )
            requests = requests[: self.config.max_requests_per_response]

        for req in requests:
            tool = req.get("tool", "")
            args = req.get("args", {})

            is_allowed, reason = self.should_allow_request(tool, args)

            if is_allowed:
                allowed.append(req)
                self.register_request(tool, args)
            else:
                blocked.append(str(reason or "Unknown blocked reason"))

        return allowed, blocked

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about tool requests.

        Returns:
            Dictionary with request statistics.
        """
        return {
            "total_requests": self.total_requests_this_run,
            "unique_requests": len(self.seen_signatures),
            "requests_by_tool": self.request_counts.copy(),
            "quota_remaining": max(
                0, self.config.max_total_requests_per_run - self.total_requests_this_run
            ),
        }

    def reset(self) -> None:
        """Reset the manager state."""
        self.seen_signatures.clear()
        self.total_requests_this_run = 0
        self.request_counts.clear()
