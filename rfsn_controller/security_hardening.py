"""Enhanced sandbox security inspired by Firecracker.

This module provides additional security hardening:
1. Seccomp-like syscall restrictions (via blocked pattern detection)
2. Jailer-inspired command isolation
3. Rate limiting to prevent resource exhaustion
4. Escape detection for shell injection attempts
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# ============================================================================
# ESCAPE DETECTION - Inspired by Firecracker's strict input validation
# ============================================================================

# Regex patterns that might indicate shell escape attempts
ESCAPE_PATTERNS: list[re.Pattern[str]] = [
    # Backtick command substitution (even if blocked by metacharacters)
    re.compile(r"`[^`]+`"),
    # $() command substitution variations
    re.compile(r"\$\([^)]+\)"),
    re.compile(r"\$\{[^}]+\}"),
    # Encoded characters that could decode to dangerous chars
    re.compile(r"\\x[0-9a-fA-F]{2}"),
    re.compile(r"\\u[0-9a-fA-F]{4}"),
    re.compile(r"%[0-9a-fA-F]{2}"),  # URL encoding
    # Null byte injection
    re.compile(r"\\0|\\x00"),
    # CRLF injection
    re.compile(r"\\r\\n|%0d%0a", re.IGNORECASE),
    # Path traversal attempts
    re.compile(r"\.\./|\.\.\\\\"),
    # Glob injection that could expand dangerously
    re.compile(r"\[!.*?\]"),  # [!pattern] negation
    # Base64 encoded command attempts
    re.compile(r"base64\s+-d", re.IGNORECASE),
    re.compile(r"echo\s+[A-Za-z0-9+/=]{20,}\s*\|"),
    # Environment variable manipulation
    re.compile(r"export\s+\w+="),
    re.compile(r"env\s+\w+="),
    # Here-doc/string injection
    re.compile(r"<<['\"]?\w+['\"]?"),
    # Process substitution
    re.compile(r"<\([^)]+\)"),
    re.compile(r">\([^)]+\)"),
]

# Arguments that could enable privilege escalation or sandbox escape
DANGEROUS_ARGUMENTS: dict[str, list[str]] = {
    "python": ["-c", "--command"],  # Could execute arbitrary code
    "python3": ["-c", "--command"],
    "node": ["-e", "--eval"],
    "ruby": ["-e"],
    "perl": ["-e"],
    "git": ["--upload-pack", "--receive-pack"],  # Could run arbitrary commands
    "tar": ["--to-command", "--checkpoint-action"],
    "find": ["-exec", "-execdir", "-ok", "-okdir"],
    "xargs": ["-I", "-i"],  # Could be used for command injection
}


def detect_escape_attempts(command: str) -> tuple[bool, str | None]:
    """Detect potential shell escape attempts in a command.

    Inspired by Firecracker's strict validation of all inputs.

    Args:
        command: The command string to analyze.

    Returns:
        (is_safe, reason) tuple. is_safe is False if escape detected.
    """
    # Check against regex patterns
    for pattern in ESCAPE_PATTERNS:
        if pattern.search(command):
            return False, f"Potential escape pattern detected: {pattern.pattern}"

    # Check for dangerous argument combinations
    parts = command.strip().split()
    if parts:
        base_cmd = parts[0]
        if base_cmd in DANGEROUS_ARGUMENTS:
            dangerous_args = DANGEROUS_ARGUMENTS[base_cmd]
            for i, arg in enumerate(parts[1:], 1):
                if arg in dangerous_args:
                    # Check if next arg looks like arbitrary code
                    if i < len(parts) and not parts[i].startswith("-"):
                        code_like = parts[i] if i < len(parts) else ""
                        if any(c in code_like for c in ["(", ")", "{", "}", "import", "eval"]):
                            return False, f"Dangerous argument detected: {arg}"

    return True, None


# ============================================================================
# RATE LIMITING - Prevent resource exhaustion attacks
# ============================================================================


@dataclass
class RateLimiter:
    """Rate limiter for sandbox commands.

    Inspired by Firecracker's resource limiting via cgroups and rlimits.
    """

    max_commands_per_minute: int = 60
    max_commands_per_hour: int = 500
    max_concurrent_commands: int = 5

    _minute_window: list[float] = field(default_factory=list)
    _hour_window: list[float] = field(default_factory=list)
    _active_commands: int = 0

    def acquire(self) -> tuple[bool, str | None]:
        """Try to acquire a rate limit slot.

        Returns:
            (allowed, reason) tuple.
        """
        now = time.time()

        # Clean up old entries
        self._minute_window = [t for t in self._minute_window if now - t < 60]
        self._hour_window = [t for t in self._hour_window if now - t < 3600]

        # Check limits
        if len(self._minute_window) >= self.max_commands_per_minute:
            return False, "Rate limit exceeded: too many commands per minute"

        if len(self._hour_window) >= self.max_commands_per_hour:
            return False, "Rate limit exceeded: too many commands per hour"

        if self._active_commands >= self.max_concurrent_commands:
            return False, "Rate limit exceeded: too many concurrent commands"

        # Record this command
        self._minute_window.append(now)
        self._hour_window.append(now)
        self._active_commands += 1

        return True, None

    def release(self) -> None:
        """Release a rate limit slot when command completes."""
        self._active_commands = max(0, self._active_commands - 1)


# ============================================================================
# SECCOMP-LIKE SYSCALL FILTERING
# ============================================================================

# Commands that might invoke dangerous syscalls
SYSCALL_RESTRICTED_COMMANDS: dict[str, str] = {
    "strace": "syscall tracing not allowed",
    "ltrace": "library tracing not allowed",
    "ptrace": "process tracing not allowed",
    "gdb": "debugger not allowed",
    "lldb": "debugger not allowed",
    "perf": "performance monitoring not allowed",
    "bpftrace": "eBPF tracing not allowed",
    "tcpdump": "packet capture not allowed",
    "wireshark": "packet capture not allowed",
    "tshark": "packet capture not allowed",
    "nmap": "network scanning not allowed",
    "masscan": "network scanning not allowed",
    "hydra": "password cracking not allowed",
    "john": "password cracking not allowed",
    "hashcat": "password cracking not allowed",
    "metasploit": "exploitation framework not allowed",
    "msfconsole": "exploitation framework not allowed",
}

# Argument patterns that could enable dangerous behaviors
SYSCALL_RESTRICTED_ARGS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"--privileged"), "privileged mode not allowed"),
    (re.compile(r"--cap-add"), "capability addition not allowed"),
    (re.compile(r"--security-opt"), "security option modification not allowed"),
    (re.compile(r"/proc/\d+/"), "access to process info not allowed"),
    (re.compile(r"/sys/kernel/"), "kernel parameter access not allowed"),
    (re.compile(r"LD_PRELOAD"), "library preloading not allowed"),
    (re.compile(r"LD_LIBRARY_PATH"), "library path modification not allowed"),
]


def check_syscall_restrictions(command: str) -> tuple[bool, str | None]:
    """Check if a command would require restricted syscalls.

    Inspired by Firecracker's seccomp filter approach.

    Args:
        command: The command string to check.

    Returns:
        (allowed, reason) tuple.
    """
    parts = command.strip().split()
    if not parts:
        return True, None

    base_cmd = parts[0]

    # Check for restricted commands
    if base_cmd in SYSCALL_RESTRICTED_COMMANDS:
        return False, SYSCALL_RESTRICTED_COMMANDS[base_cmd]

    # Check for restricted argument patterns
    for pattern, reason in SYSCALL_RESTRICTED_ARGS:
        if pattern.search(command):
            return False, reason

    return True, None


# ============================================================================
# COMPREHENSIVE SECURITY CHECK
# ============================================================================


def security_check(command: str, rate_limiter: RateLimiter | None = None) -> tuple[bool, str | None]:
    """Perform comprehensive security checks on a command.

    Combines:
    - Escape attempt detection
    - Syscall restriction checking
    - Rate limiting (if limiter provided)

    Args:
        command: The command to check.
        rate_limiter: Optional rate limiter instance.

    Returns:
        (allowed, reason) tuple.
    """
    # Check for escape attempts
    is_safe, reason = detect_escape_attempts(command)
    if not is_safe:
        return False, reason

    # Check syscall restrictions
    allowed, reason = check_syscall_restrictions(command)
    if not allowed:
        return False, reason

    # Check rate limits
    if rate_limiter:
        allowed, reason = rate_limiter.acquire()
        if not allowed:
            return False, reason

    return True, None


# ============================================================================
# JAILER-INSPIRED ISOLATION OPTIONS
# ============================================================================


@dataclass
class IsolationConfig:
    """Configuration for sandbox isolation.

    Inspired by Firecracker jailer's isolation model.
    """

    # Resource limits (matching Firecracker jailer --resource-limit)
    max_file_size_bytes: int = 250_000_000  # 250MB
    max_open_files: int = 1024

    # Network control
    network_enabled: bool = True
    allowed_hosts: set[str] = field(default_factory=set)  # Empty = all allowed

    # Filesystem isolation
    read_only_root: bool = False
    tmpfs_size_mb: int = 512

    # Process limits
    max_pids: int = 256
    max_cpu_percent: float = 200.0  # 2 CPUs
    max_memory_mb: int = 4096

    def to_docker_args(self) -> list[str]:
        """Convert to Docker run arguments."""
        args = []

        # Resource limits
        args.extend([
            f"--cpus={self.max_cpu_percent / 100}",
            f"--memory={self.max_memory_mb}m",
            f"--pids-limit={self.max_pids}",
        ])

        # Filesystem
        if self.read_only_root:
            args.append("--read-only")
            args.append(f"--tmpfs=/tmp:rw,noexec,nosuid,size={self.tmpfs_size_mb}m")

        # Network
        if not self.network_enabled:
            args.append("--network=none")

        return args
