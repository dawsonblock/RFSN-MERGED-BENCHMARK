"""Multi-language stack trace parsers.
from __future__ import annotations

Extracts file paths, line numbers, and error context from stack traces
across different languages (Python, Node, Java, Go, Rust).
"""

import re
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported languages for trace parsing."""

    PYTHON = "python"
    NODE = "node"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class StackFrame:
    """A single stack frame."""

    filepath: str
    line_number: int | None
    function_name: str
    language: Language


@dataclass
class ParsedTrace:
    """Result of parsing a stack trace."""

    frames: list[StackFrame]
    error_type: str | None
    error_message: str
    language: Language


class TraceParser:
    """Parses stack traces from multiple languages."""

    # Python traceback patterns
    PYTHON_PATTERNS = [
        r'File "([^"]+)", line (\d+), in ([\w<>]+)',
    ]

    # Node.js stack trace patterns
    NODE_PATTERNS = [
        r"at ([\w\.]+) \(([^:]+):(\d+):\d+\)",
        r"at ([^:]+):(\d+):\d+",
        r"at ([\w\.]+) \(([^)]+)\)",
    ]

    # Java exception patterns
    JAVA_PATTERNS = [
        r"at ([\w.$]+)\(([^:]+\.java):(\d+)\)",
        r"Caused by: ([\w.$]+):",
    ]

    # Go panic patterns
    GO_PATTERNS = [
        r"([\w/]+\.go):(\d+)",
        r"created by ([\w./]+)",
    ]

    # Rust panic patterns
    RUST_PATTERNS = [
        r"   \d+: ([\w:]+)\n\s*at ([\w/]+\.rs):(\d+):(\d+)",
        r"at ([\w/]+\.rs):(\d+):(\d+)",
    ]

    def __init__(self):
        """Initialize the trace parser."""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for each language."""
        self.python_regexes = [re.compile(p) for p in self.PYTHON_PATTERNS]
        self.node_regexes = [re.compile(p) for p in self.NODE_PATTERNS]
        self.java_regexes = [re.compile(p) for p in self.JAVA_PATTERNS]
        self.go_regexes = [re.compile(p) for p in self.GO_PATTERNS]
        # Rust patterns need MULTILINE flag for \n matching
        self.rust_regexes = [re.compile(p, re.MULTILINE) for p in self.RUST_PATTERNS]

    def detect_language(self, trace: str) -> Language:
        """Detect the language of a stack trace.

        Args:
            trace: Stack trace string.

        Returns:
            Detected Language.
        """
        trace_lower = trace.lower()

        # Python indicators (most specific first)
        if "traceback (most recent call last)" in trace_lower:
            return Language.PYTHON

        # Rust indicators (most specific)
        if "panicked at" in trace_lower or "thread '" in trace_lower:
            return Language.RUST

        # Go indicators
        if "panic:" in trace_lower or "goroutine" in trace_lower:
            return Language.GO

        # Java indicators (check for .java: before "at ")
        if ".java:" in trace_lower or "exception" in trace_lower:
            return Language.JAVA

        # Node.js indicators (most generic, check last)
        if ".js:" in trace_lower or "node:" in trace_lower:
            return Language.NODE

        return Language.UNKNOWN

    def parse(self, trace: str, language: Language | None = None) -> ParsedTrace:
        """Parse a stack trace.

        Args:
            trace: Stack trace string.
            language: Optional language hint (auto-detected if None).

        Returns:
            ParsedTrace with frames and error info.
        """
        if language is None:
            language = self.detect_language(trace)

        if language == Language.PYTHON:
            return self._parse_python(trace)
        elif language == Language.NODE:
            return self._parse_node(trace)
        elif language == Language.JAVA:
            return self._parse_java(trace)
        elif language == Language.GO:
            return self._parse_go(trace)
        elif language == Language.RUST:
            return self._parse_rust(trace)
        else:
            return ParsedTrace(
                frames=[],
                error_type=None,
                error_message=trace.split("\n")[0],
                language=Language.UNKNOWN,
            )

    def _parse_python(self, trace: str) -> ParsedTrace:
        """Parse Python traceback."""
        frames = []
        error_type = None
        error_message = ""

        lines = trace.split("\n")

        for i, line in enumerate(lines):
            # Extract error type and message (last line with colon)
            if i == len(lines) - 1 and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    error_type = parts[0].strip()
                    error_message = parts[1].strip()

            # Extract frames - check for File pattern
            for regex in self.python_regexes:
                match = regex.search(line)
                if match:
                    filepath = match.group(1)
                    line_number = int(match.group(2))
                    function_name = match.group(3)

                    frames.append(
                        StackFrame(
                            filepath=filepath,
                            line_number=line_number,
                            function_name=function_name,
                            language=Language.PYTHON,
                        )
                    )
                    break  # Only match one pattern per line

        return ParsedTrace(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            language=Language.PYTHON,
        )

    def _parse_node(self, trace: str) -> ParsedTrace:
        """Parse Node.js stack trace."""
        frames = []
        error_type = None
        error_message = ""

        lines = trace.split("\n")

        for line in lines:
            # Extract error type and message (first line)
            if ":" in line and not line.strip().startswith("at"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    error_type = parts[0].strip()
                    error_message = parts[1].strip()

            # Extract frames
            for regex in self.node_regexes:
                match = regex.search(line)
                if match:
                    groups = match.groups()

                    if len(groups) == 3:
                        # Pattern: at function (file:line:col)
                        function_name = groups[0]
                        filepath = groups[1]
                        line_number = int(groups[2])
                    elif len(groups) == 2:
                        # Pattern: at file:line:col
                        filepath = groups[0]
                        line_number = int(groups[1])
                        function_name = "<anonymous>"
                    else:
                        continue

                    frames.append(
                        StackFrame(
                            filepath=filepath,
                            line_number=line_number,
                            function_name=function_name,
                            language=Language.NODE,
                        )
                    )
                    break

        return ParsedTrace(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            language=Language.NODE,
        )

    def _parse_java(self, trace: str) -> ParsedTrace:
        """Parse Java exception stack trace."""
        frames = []
        error_type = None
        error_message = ""

        lines = trace.split("\n")

        for line in lines:
            # Extract error type and message (first line with exception)
            if "Exception" in line or "Error" in line:
                # Java exceptions typically look like:
                # "Exception in thread \"main\" java.lang.NullPointerException: message"
                # We need to extract the actual exception class name
                if ":" in line:
                    # Split on first colon to separate exception from message
                    parts = line.split(":", 1)
                    # The exception type is usually the last word before the colon
                    # e.g., "java.lang.NullPointerException" from "... NullPointerException:"
                    before_colon = parts[0].strip()
                    # Extract the last word as the exception type
                    words = before_colon.split()
                    if words:
                        error_type = words[-1]
                    error_message = parts[1].strip()
                else:
                    # No colon, just take the last word as exception type
                    words = line.split()
                    if words:
                        error_type = words[-1]

            # Extract frames
            for regex in self.java_regexes:
                match = regex.search(line)
                if match:
                    groups = match.groups()

                    if len(groups) == 3:
                        # Pattern: at Class.method (File.java:line)
                        class_method = groups[0]
                        filepath = groups[1]
                        line_number = int(groups[2])

                        frames.append(
                            StackFrame(
                                filepath=filepath,
                                line_number=line_number,
                                function_name=class_method,
                                language=Language.JAVA,
                            )
                        )
                    break

        return ParsedTrace(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            language=Language.JAVA,
        )

    def _parse_go(self, trace: str) -> ParsedTrace:
        """Parse Go panic trace."""
        frames = []
        error_type = "panic"
        error_message = ""

        lines = trace.split("\n")

        for line in lines:
            # Extract panic message
            if line.strip().startswith("panic:"):
                error_message = line.strip()[6:].strip()

            # Extract frames
            for regex in self.go_regexes:
                match = regex.search(line)
                if match:
                    groups = match.groups()

                    if len(groups) == 2:
                        # Pattern: file.go:line
                        filepath = groups[0]
                        line_number = int(groups[1])
                        function_name = "<unknown>"

                        frames.append(
                            StackFrame(
                                filepath=filepath,
                                line_number=line_number,
                                function_name=function_name,
                                language=Language.GO,
                            )
                        )
                    break

        return ParsedTrace(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            language=Language.GO,
        )

    def _parse_rust(self, trace: str) -> ParsedTrace:
        """Parse Rust panic trace."""
        frames = []
        error_type = "panic"
        error_message = ""

        # Extract panic message
        if "panicked at" in trace:
            parts = trace.split("'", 1)
            if len(parts) == 2:
                error_message = parts[1].split("'")[0]

        # Extract frames using regex patterns on entire trace
        for regex in self.rust_regexes:
            matches = regex.findall(trace)
            for match in matches:
                if len(match) == 4:
                    # Pattern with function name, filepath, line, column
                    function_name = match[0]
                    filepath = match[1]
                    line_number = int(match[2])

                    frames.append(
                        StackFrame(
                            filepath=filepath,
                            line_number=line_number,
                            function_name=function_name,
                            language=Language.RUST,
                        )
                    )
                elif len(match) == 3:
                    # Pattern without column number
                    function_name = match[0]
                    filepath = match[1]
                    line_number = int(match[2])

                    frames.append(
                        StackFrame(
                            filepath=filepath,
                            line_number=line_number,
                            function_name=function_name,
                            language=Language.RUST,
                        )
                    )

        return ParsedTrace(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            language=Language.RUST,
        )

    def extract_files_to_examine(self, trace: str) -> list[str]:
        """Extract unique file paths from a stack trace.

        Args:
            trace: Stack trace string.

        Returns:
            List of unique file paths.
        """
        parsed = self.parse(trace)
        files = set()

        for frame in parsed.frames:
            files.add(frame.filepath)

        return sorted(list(files))
