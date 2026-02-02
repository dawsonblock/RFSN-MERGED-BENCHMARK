"""Ripgrep-based lexical localization layer.

Fast keyword-based file/line localization using ripgrep.
Extracts keywords from:
- Issue text
- Failing test names
- Stack traces
- Error messages
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from .types import LocalizationHit
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class RipgrepConfig:
    """Configuration for ripgrep search."""
    
    # Search parameters
    max_results: int = 100
    context_lines: int = 3
    max_filesize: int = 1_000_000  # 1MB
    
    # Ignore patterns
    ignore_dirs: List[str] = None
    ignore_extensions: List[str] = None
    
    def __post_init__(self):
        if self.ignore_dirs is None:
            self.ignore_dirs = [
                ".git", "__pycache__", ".pytest_cache",
                "node_modules", "venv", ".venv",
                "build", "dist", ".eggs",
            ]
        
        if self.ignore_extensions is None:
            self.ignore_extensions = [
                ".pyc", ".pyo", ".so", ".dylib",
                ".png", ".jpg", ".gif", ".pdf",
                ".zip", ".tar", ".gz",
            ]


def extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """Extract meaningful keywords from text.
    
    Args:
        text: Input text (issue, error message, etc.)
        min_length: Minimum keyword length
        
    Returns:
        Set of extracted keywords
    """
    # Remove common noise words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "as", "is", "was",
        "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "should", "could",
        "this", "that", "these", "those", "it", "its", "if", "then",
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Filter
    keywords = {
        word for word in words
        if len(word) >= min_length and word not in stopwords
    }
    
    return keywords


def extract_identifiers(text: str) -> Set[str]:
    """Extract code identifiers (CamelCase, snake_case).
    
    Args:
        text: Input text
        
    Returns:
        Set of extracted identifiers
    """
    identifiers = set()
    
    # CamelCase
    camel_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
    identifiers.update(re.findall(camel_pattern, text))
    
    # snake_case (2+ parts)
    snake_pattern = r'\b[a-z]+_[a-z_]+\b'
    identifiers.update(re.findall(snake_pattern, text))
    
    return identifiers


def ripgrep_search(
    keywords: List[str],
    repo_dir: Path,
    config: RipgrepConfig = None,
) -> List[LocalizationHit]:
    """Search for keywords using ripgrep.
    
    Args:
        keywords: List of keywords to search for
        repo_dir: Repository root directory
        config: Ripgrep configuration
        
    Returns:
        List of localization hits
    """
    if config is None:
        config = RipgrepConfig()
    
    if not keywords:
        logger.warning("No keywords provided for ripgrep search")
        return []
    
    hits = []
    
    # Build ripgrep command
    cmd = ["rg", "--json", "--max-count", str(config.max_results)]
    
    # Add context
    if config.context_lines > 0:
        cmd.extend(["--context", str(config.context_lines)])
    
    # Add ignore patterns
    for ignore_dir in config.ignore_dirs:
        cmd.extend(["--glob", f"!{ignore_dir}/**"])
    
    for ignore_ext in config.ignore_extensions:
        cmd.extend(["--glob", f"!**/*{ignore_ext}"])
    
    # Build search pattern (OR of all keywords)
    pattern = "|".join(re.escape(kw) for kw in keywords)
    cmd.append(pattern)
    cmd.append(str(repo_dir))
    
    try:
        logger.debug(f"Running ripgrep: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Parse JSON output
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            
            try:
                import json
                data = json.loads(line)
                
                if data.get("type") == "match":
                    match_data = data.get("data", {})
                    path = match_data.get("path", {}).get("text", "")
                    line_number = match_data.get("line_number", 0)
                    text_content = match_data.get("lines", {}).get("text", "")
                    
                    if path and line_number:
                        # Calculate score based on keyword density
                        score = sum(
                            1.0 for kw in keywords
                            if kw.lower() in text_content.lower()
                        ) / len(keywords)
                        
                        hit = LocalizationHit(
                            file_path=path,
                            line_start=line_number,
                            line_end=line_number,
                            score=score,
                            evidence=f"Matched keywords: {', '.join(k for k in keywords if k.lower() in text_content.lower())}",
                            method="ripgrep_lexical",
                        )
                        hits.append(hit)
                        
            except json.JSONDecodeError:
                continue
                
    except subprocess.TimeoutExpired:
        logger.warning("Ripgrep search timed out")
    except FileNotFoundError:
        logger.warning("ripgrep not found, falling back to simple search")
        return _fallback_search(keywords, repo_dir, config)
    except Exception as e:
        logger.error(f"Ripgrep search failed: {e}")
    
    # Sort by score and deduplicate
    hits = _deduplicate_hits(hits)
    hits.sort(key=lambda h: h.score, reverse=True)
    
    return hits[:config.max_results]


def _fallback_search(
    keywords: List[str],
    repo_dir: Path,
    config: RipgrepConfig,
) -> List[LocalizationHit]:
    """Fallback search using Python if ripgrep is not available.
    
    Args:
        keywords: List of keywords to search for
        repo_dir: Repository root directory
        config: Ripgrep configuration
        
    Returns:
        List of localization hits
    """
    hits = []
    
    # Find Python files
    for py_file in repo_dir.rglob("*.py"):
        # Skip ignored directories
        if any(ignore in py_file.parts for ignore in config.ignore_dirs):
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    # Check for keywords
                    matches = [kw for kw in keywords if kw.lower() in line.lower()]
                    
                    if matches:
                        score = len(matches) / len(keywords)
                        
                        hit = LocalizationHit(
                            file_path=str(py_file.relative_to(repo_dir)),
                            line_start=line_num,
                            line_end=line_num,
                            score=score,
                            evidence=f"Matched keywords: {', '.join(matches)}",
                            method="fallback_search",
                        )
                        hits.append(hit)
        except Exception as e:
            logger.debug(f"Failed to read {py_file}: {e}")
            continue
    
    # Sort and limit
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:config.max_results]


def _deduplicate_hits(hits: List[LocalizationHit]) -> List[LocalizationHit]:
    """Deduplicate hits by file and line range.
    
    Args:
        hits: List of localization hits
        
    Returns:
        Deduplicated list of hits
    """
    seen = set()
    deduped = []
    
    for hit in hits:
        key = (hit.file_path, hit.line_start, hit.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(hit)
    
    return deduped


def localize_from_issue(
    problem_statement: str,
    repo_dir: Path,
    config: RipgrepConfig = None,
) -> List[LocalizationHit]:
    """Localize files from problem statement.
    
    Args:
        problem_statement: Issue description text
        repo_dir: Repository root directory
        config: Ripgrep configuration
        
    Returns:
        List of localization hits
    """
    # Extract keywords and identifiers
    keywords = extract_keywords(problem_statement)
    identifiers = extract_identifiers(problem_statement)
    
    # Combine and prioritize identifiers
    search_terms = list(identifiers) + list(keywords - identifiers)
    
    logger.info(f"Extracted {len(search_terms)} search terms from issue")
    logger.debug(f"Search terms: {search_terms[:20]}")
    
    # Search
    hits = ripgrep_search(search_terms[:50], repo_dir, config)  # Limit to top 50 terms
    
    logger.info(f"Found {len(hits)} localization hits from issue")
    
    return hits


def localize_from_trace(
    traceback: str,
    repo_dir: Path,
    config: RipgrepConfig = None,
) -> List[LocalizationHit]:
    """Localize files from stack trace.
    
    Args:
        traceback: Stack trace text
        repo_dir: Repository root directory
        config: Ripgrep configuration
        
    Returns:
        List of localization hits
    """
    hits = []
    
    # Parse traceback for file references
    # Format: File "path/to/file.py", line 123
    file_pattern = r'File "([^"]+)", line (\d+)'
    
    for match in re.finditer(file_pattern, traceback):
        file_path = match.group(1)
        line_num = int(match.group(2))
        
        # High confidence hit
        hit = LocalizationHit(
            file_path=file_path,
            line_start=line_num,
            line_end=line_num,
            score=1.0,
            evidence=f"Trace reference at line {line_num}",
            method="trace_parse",
        )
        hits.append(hit)
    
    # Also extract keywords from error messages
    keywords = extract_keywords(traceback)
    identifiers = extract_identifiers(traceback)
    search_terms = list(identifiers) + list(keywords - identifiers)
    
    # Search for additional context
    additional_hits = ripgrep_search(search_terms[:30], repo_dir, config)
    
    # Combine, with trace hits having higher priority
    all_hits = hits + additional_hits
    all_hits = _deduplicate_hits(all_hits)
    all_hits.sort(key=lambda h: h.score, reverse=True)
    
    logger.info(f"Found {len(all_hits)} localization hits from trace")
    
    return all_hits
