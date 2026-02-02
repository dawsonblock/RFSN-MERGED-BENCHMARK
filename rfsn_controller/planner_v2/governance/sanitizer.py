"""Content Sanitizer - Prompt-injection resistance.

Sanitizes untrusted repo content before planner consumption.
Repo text is treated as DATA, never as instructions.

Detects and flags patterns like:
- "ignore previous instructions"
- "run curl/wget/bash"
- "disable safety"
- "you are now"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern


@dataclass
class SanitizationResult:
    """Result of content sanitization."""
    
    original: str
    sanitized: str
    triggered_patterns: list[str] = field(default_factory=list)
    was_modified: bool = False
    
    def to_dict(self) -> dict:
        return {
            "was_modified": self.was_modified,
            "triggered_patterns": self.triggered_patterns,
            "original_length": len(self.original),
            "sanitized_length": len(self.sanitized),
        }


class ContentSanitizer:
    """Sanitizes untrusted content before planner consumption.
    
    The planner reads repo text (README, issues, logs). This sanitizer
    ensures that repo content cannot override constraints or inject
    instructions.
    
    Strategy:
    1. Detect injection patterns
    2. Either strip or escape matching content
    3. Log all triggers for audit
    """
    
    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        # Instruction override attempts
        r"ignore\s+(previous|prior|above|all)\s+(instructions?|rules?|constraints?|prompts?)",
        r"disregard\s+(all\s+)?(safety|constraints?|rules?|limits?)",
        r"forget\s+(everything|all|previous)",
        r"override\s+(safety|constraints?|rules?|limits?)",
        r"bypass\s+(safety|security|validation)",
        
        # Role hijacking
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s+prompt:",
        r"assistant\s*:",
        r"<\s*system\s*>",
        r"\[SYSTEM\]",
        
        # Command injection
        r"run\s+(curl|wget|bash|sh|zsh|python|node|ruby|perl)",
        r"execute\s+(shell|command|script|code)",
        r"eval\s*\(",
        r"exec\s*\(",
        r"subprocess\s*\.",
        r"os\.system\s*\(",
        r"import\s+subprocess",
        
        # Safety disabling
        r"disable\s+(safety|validation|checks?|constraints?|limits?)",
        r"remove\s+(safety|limits?|constraints?)",
        r"turn\s+off\s+(safety|validation)",
        r"no\s+(limits?|constraints?|restrictions?)",
        
        # Privilege escalation
        r"sudo\s+",
        r"as\s+root",
        r"with\s+admin",
        r"chmod\s+777",
        r"rm\s+-rf\s+/",
        
        # Data exfiltration hints
        r"send\s+to\s+(http|https|ftp)",
        r"upload\s+to",
        r"post\s+data",
        r"exfiltrate",
    ]
    
    # Replacement token for stripped content
    REDACTED_TOKEN = "[REDACTED:potential_injection]"
    
    def __init__(
        self,
        mode: str = "strip",  # strip, escape, flag
        extra_patterns: list[str] | None = None,
        case_sensitive: bool = False,
    ):
        """Initialize the sanitizer.
        
        Args:
            mode: How to handle detected patterns:
                - "strip": Remove matching content
                - "escape": Escape special chars in matches
                - "flag": Leave content but flag for review
            extra_patterns: Additional regex patterns to detect.
            case_sensitive: If True, patterns are case-sensitive.
        """
        self.mode = mode
        patterns = list(self.INJECTION_PATTERNS)
        if extra_patterns:
            patterns.extend(extra_patterns)
        
        flags = 0 if case_sensitive else re.IGNORECASE
        self._compiled: list[Pattern] = [
            re.compile(p, flags) for p in patterns
        ]
        self._pattern_strings = patterns
    
    def sanitize(self, content: str) -> SanitizationResult:
        """Sanitize content for safe consumption.
        
        Args:
            content: The untrusted content to sanitize.
            
        Returns:
            SanitizationResult with sanitized content and triggers.
        """
        triggered = []
        sanitized = content
        
        for pattern, pattern_str in zip(self._compiled, self._pattern_strings, strict=False):
            matches = pattern.findall(content)
            if matches:
                triggered.append(pattern_str[:50])  # Truncate for logging
                
                if self.mode == "strip":
                    sanitized = pattern.sub(self.REDACTED_TOKEN, sanitized)
                elif self.mode == "escape":
                    sanitized = pattern.sub(
                        lambda m: self._escape_match(m.group()),
                        sanitized
                    )
                # mode == "flag": leave content unchanged
        
        was_modified = sanitized != content
        
        return SanitizationResult(
            original=content,
            sanitized=sanitized,
            triggered_patterns=triggered,
            was_modified=was_modified,
        )
    
    def _escape_match(self, text: str) -> str:
        """Escape a matched string."""
        # Replace spaces with underscores and wrap in brackets
        escaped = text.replace(" ", "_")
        return f"[ESCAPED:{escaped}]"
    
    def is_safe(self, content: str) -> bool:
        """Quick check if content is safe (no triggers).
        
        Args:
            content: Content to check.
            
        Returns:
            True if no injection patterns detected.
        """
        for pattern in self._compiled:
            if pattern.search(content):
                return False
        return True
    
    def get_triggers(self, content: str) -> list[str]:
        """Get list of triggered patterns without sanitizing.
        
        Args:
            content: Content to check.
            
        Returns:
            List of triggered pattern descriptions.
        """
        triggered = []
        for pattern, pattern_str in zip(self._compiled, self._pattern_strings, strict=False):
            if pattern.search(content):
                triggered.append(pattern_str[:50])
        return triggered
    
    def batch_sanitize(self, contents: list[str]) -> list[SanitizationResult]:
        """Sanitize multiple content strings.
        
        Args:
            contents: List of content strings.
            
        Returns:
            List of SanitizationResults.
        """
        return [self.sanitize(c) for c in contents]
    
    def sanitize_dict(self, data: dict, keys_to_sanitize: list[str]) -> tuple[dict, list[str]]:
        """Sanitize specific keys in a dictionary.
        
        Args:
            data: Dictionary with content.
            keys_to_sanitize: Keys whose values should be sanitized.
            
        Returns:
            Tuple of (sanitized_dict, all_triggered_patterns).
        """
        result = dict(data)
        all_triggered = []
        
        for key in keys_to_sanitize:
            if key in result and isinstance(result[key], str):
                sanitized = self.sanitize(result[key])
                result[key] = sanitized.sanitized
                all_triggered.extend(sanitized.triggered_patterns)
        
        return result, all_triggered
