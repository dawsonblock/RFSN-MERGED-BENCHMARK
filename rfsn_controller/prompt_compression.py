"""Prompt compression utilities for faster LLM inference.

Reduces token count while preserving semantic meaning.
"""

from __future__ import annotations

import re


def compress_prompt(
    prompt: str,
    *,
    max_tokens: int | None = None,
    aggressive: bool = False,
) -> tuple[str, dict[str, int]]:
    """Compress a prompt to reduce token count.
    
    Techniques used:
    1. Remove redundant whitespace
    2. Abbreviate common patterns
    3. Deduplicate repeated content
    4. Remove comments from code blocks
    
    Args:
        prompt: The original prompt text.
        max_tokens: Optional maximum token estimate (4 chars ≈ 1 token).
        aggressive: Use more aggressive compression (may lose some context).
        
    Returns:
        (compressed_prompt, stats) tuple with compression statistics.
    """
    original_len = len(prompt)
    
    # Track stats
    stats = {
        "original_chars": original_len,
        "original_est_tokens": original_len // 4,
    }
    
    compressed = prompt
    
    # 1. Normalize whitespace (always safe)
    compressed = re.sub(r'\n{3,}', '\n\n', compressed)  # Limit consecutive newlines
    compressed = re.sub(r'[ \t]+', ' ', compressed)  # Collapse horizontal whitespace
    compressed = re.sub(r' +\n', '\n', compressed)  # Remove trailing spaces
    
    # 2. Remove code comments (if aggressive)
    if aggressive:
        # Python comments
        compressed = re.sub(r'#[^\n]*\n', '\n', compressed)
        # JS/C-style line comments
        compressed = re.sub(r'//[^\n]*\n', '\n', compressed)
        # Multi-line comments
        compressed = re.sub(r'/\*.*?\*/', '', compressed, flags=re.DOTALL)
    
    # 3. Common abbreviations (safe)
    abbreviations = [
        (r'\bfunction\b', 'fn'),
        (r'\breturn\b', 'ret'),
        (r'\bimport\b', 'imp'),
        (r'\bcontinue\b', 'cont'),
        (r'\bbreakpoint\b', 'bp'),
    ]
    
    if aggressive:
        for pattern, replacement in abbreviations:
            compressed = re.sub(pattern, replacement, compressed)
    
    # 4. Deduplicate repeated error messages
    lines = compressed.split('\n')
    seen_errors: dict[str, int] = {}
    deduped_lines: list[str] = []
    
    for line in lines:
        # Check if it's an error/traceback line
        if 'Error' in line or 'Exception' in line or 'Traceback' in line:
            line_hash = hash(line.strip())
            if line_hash in seen_errors:
                seen_errors[line_hash] += 1
                continue  # Skip duplicate
            seen_errors[line_hash] = 1
        deduped_lines.append(line)
    
    compressed = '\n'.join(deduped_lines)
    
    # 5. Add summary for repeated errors
    for line_hash, count in seen_errors.items():
        if count > 1:
            compressed += f"\n[... {count-1} similar error(s) omitted ...]"
            break  # Only add once
    
    # 6. Truncate if max_tokens specified
    if max_tokens:
        max_chars = max_tokens * 4
        if len(compressed) > max_chars:
            # Keep start and end, remove middle
            keep_start = max_chars // 2
            keep_end = max_chars // 2 - 100
            compressed = (
                compressed[:keep_start] +
                f"\n\n[... {len(compressed) - max_chars} characters truncated ...]\n\n" +
                compressed[-keep_end:]
            )
    
    # Final stats
    stats["compressed_chars"] = len(compressed)
    stats["compressed_est_tokens"] = len(compressed) // 4
    stats["reduction_percent"] = round(
        (1 - len(compressed) / original_len) * 100, 1
    ) if original_len > 0 else 0
    
    return compressed, stats


def compress_file_content(content: str, filename: str) -> str:
    """Compress file content based on file type.
    
    Args:
        content: File content.
        filename: Filename to determine type.
        
    Returns:
        Compressed content.
    """
    # Detect file type
    if filename.endswith('.py'):
        # Remove docstrings for very long files
        if len(content) > 10000:
            content = re.sub(r'""".*?"""', '"""..."""', content, flags=re.DOTALL)
            content = re.sub(r"'''.*?'''", "'''...'''", content, flags=re.DOTALL)
    
    elif filename.endswith(('.js', '.ts', '.jsx', '.tsx')):
        # Remove JSDoc comments for very long files
        if len(content) > 10000:
            content = re.sub(r'/\*\*.*?\*/', '/**...*/', content, flags=re.DOTALL)
    
    elif filename.endswith('.json'):
        # Minify JSON
        try:
            import json
            parsed = json.loads(content)
            content = json.dumps(parsed, separators=(',', ':'))
        except Exception:
            pass
    
    return content


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 4 chars ≈ 1 token).
    
    Args:
        text: Text to estimate.
        
    Returns:
        Estimated token count.
    """
    # More accurate estimation considering whitespace and punctuation
    words = len(text.split())
    chars = len(text)
    
    # Average of word-based and char-based estimates
    word_estimate = words * 1.3
    char_estimate = chars / 4
    
    return int((word_estimate + char_estimate) / 2)
