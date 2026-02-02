"""Symbol index for code structure analysis.

Builds an index of:
- Function/class definitions
- Import relationships
- Call graphs (simple heuristic)
- Symbol references

Uses ctags if available, falls back to regex patterns.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .types import LocalizationHit
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class Symbol:
    """A code symbol (function, class, method)."""
    
    name: str
    kind: str  # function, class, method, variable
    file_path: str
    line_number: int
    signature: Optional[str] = None


@dataclass
class ImportRelation:
    """An import relationship."""
    
    from_file: str
    to_module: str
    imported_names: List[str] = field(default_factory=list)
    line_number: int = 0


class SymbolIndex:
    """Index of code symbols and relationships."""
    
    def __init__(self):
        """Initialize symbol index."""
        self.symbols: Dict[str, List[Symbol]] = defaultdict(list)
        self.imports: List[ImportRelation] = []
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def build(self, repo_dir: Path) -> None:
        """Build symbol index for repository.
        
        Args:
            repo_dir: Repository root directory
        """
        logger.info(f"Building symbol index for {repo_dir}")
        
        # Try ctags first
        if self._try_ctags(repo_dir):
            logger.info("Built symbol index using ctags")
        else:
            logger.info("Falling back to regex-based symbol extraction")
            self._fallback_extraction(repo_dir)
        
        # Extract imports and build call graph
        self._extract_imports(repo_dir)
        self._build_call_graph(repo_dir)
        
        logger.info(f"Indexed {len(self.symbols)} symbols, "
                   f"{len(self.imports)} imports")
    
    def find_symbol(self, name: str) -> List[Symbol]:
        """Find symbols by name.
        
        Args:
            name: Symbol name
            
        Returns:
            List of matching symbols
        """
        return self.symbols.get(name, [])
    
    def find_callers(self, symbol_name: str) -> Set[str]:
        """Find potential callers of a symbol.
        
        Args:
            symbol_name: Symbol to find callers for
            
        Returns:
            Set of file paths that may call this symbol
        """
        callers = set()
        
        # Find files that import the module containing this symbol
        symbol_defs = self.find_symbol(symbol_name)
        
        for sym_def in symbol_defs:
            # Find imports of this file
            for import_rel in self.imports:
                if (symbol_name in import_rel.imported_names or
                    sym_def.file_path.replace("/", ".").replace(".py", "") in import_rel.to_module):
                    callers.add(import_rel.from_file)
        
        return callers
    
    def find_related_files(self, file_path: str, max_depth: int = 2) -> Set[str]:
        """Find files related through imports.
        
        Args:
            file_path: Starting file path
            max_depth: Maximum import depth to traverse
            
        Returns:
            Set of related file paths
        """
        related = {file_path}
        frontier = {file_path}
        
        for _ in range(max_depth):
            next_frontier = set()
            
            for current_file in frontier:
                # Find imports from this file
                for import_rel in self.imports:
                    if import_rel.from_file == current_file:
                        # Try to resolve module to file
                        module_file = self._resolve_module(import_rel.to_module)
                        if module_file and module_file not in related:
                            next_frontier.add(module_file)
                            related.add(module_file)
                
                # Find files that import this file
                for import_rel in self.imports:
                    module_from_file = current_file.replace("/", ".").replace(".py", "")
                    if module_from_file in import_rel.to_module:
                        if import_rel.from_file not in related:
                            next_frontier.add(import_rel.from_file)
                            related.add(import_rel.from_file)
            
            frontier = next_frontier
            if not frontier:
                break
        
        return related
    
    def localize_by_symbol(self, symbol_name: str) -> List[LocalizationHit]:
        """Localize by symbol name.
        
        Args:
            symbol_name: Symbol to localize
            
        Returns:
            List of localization hits
        """
        hits = []
        
        # Direct symbol definitions
        for symbol in self.find_symbol(symbol_name):
            hit = LocalizationHit(
                file_path=symbol.file_path,
                line_start=symbol.line_number,
                line_end=symbol.line_number,
                score=1.0,
                evidence=f"Symbol definition: {symbol.kind} {symbol.name}",
                method="symbol_definition",
            )
            hits.append(hit)
        
        # Files that call this symbol
        callers = self.find_callers(symbol_name)
        for caller_file in callers:
            hit = LocalizationHit(
                file_path=caller_file,
                line_start=1,
                line_end=1,
                score=0.7,
                evidence=f"Imports/calls {symbol_name}",
                method="symbol_caller",
            )
            hits.append(hit)
        
        return hits
    
    def _try_ctags(self, repo_dir: Path) -> bool:
        """Try to build index using ctags.
        
        Args:
            repo_dir: Repository root directory
            
        Returns:
            True if successful
        """
        try:
            # Run ctags
            cmd = [
                "ctags",
                "-R",
                "--fields=+n+S",
                "--output-format=json",
                "--languages=Python",
                str(repo_dir),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                return False
            
            # Parse JSON output
            import json
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    symbol = Symbol(
                        name=data.get("name", ""),
                        kind=data.get("kind", ""),
                        file_path=str(Path(data.get("path", "")).relative_to(repo_dir)),
                        line_number=data.get("line", 0),
                        signature=data.get("signature"),
                    )
                    
                    self.symbols[symbol.name].append(symbol)
                    
                except (json.JSONDecodeError, ValueError):
                    continue
            
            return len(self.symbols) > 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _fallback_extraction(self, repo_dir: Path) -> None:
        """Fallback regex-based symbol extraction.
        
        Args:
            repo_dir: Repository root directory
        """
        # Patterns for Python
        class_pattern = re.compile(r'^\s*class\s+(\w+)')
        func_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(')
        
        for py_file in repo_dir.rglob("*.py"):
            if any(part in [".git", "__pycache__", "venv", ".venv"]
                   for part in py_file.parts):
                continue
            
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    rel_path = str(py_file.relative_to(repo_dir))
                    
                    for line_num, line in enumerate(f, 1):
                        # Check for class
                        match = class_pattern.match(line)
                        if match:
                            symbol = Symbol(
                                name=match.group(1),
                                kind="class",
                                file_path=rel_path,
                                line_number=line_num,
                            )
                            self.symbols[symbol.name].append(symbol)
                        
                        # Check for function
                        match = func_pattern.match(line)
                        if match:
                            symbol = Symbol(
                                name=match.group(1),
                                kind="function",
                                file_path=rel_path,
                                line_number=line_num,
                            )
                            self.symbols[symbol.name].append(symbol)
                            
            except Exception as e:
                logger.debug(f"Failed to extract symbols from {py_file}: {e}")
    
    def _extract_imports(self, repo_dir: Path) -> None:
        """Extract import relationships.
        
        Args:
            repo_dir: Repository root directory
        """
        import_pattern = re.compile(
            r'^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w\s,]+)'
        )
        
        for py_file in repo_dir.rglob("*.py"):
            if any(part in [".git", "__pycache__", "venv", ".venv"]
                   for part in py_file.parts):
                continue
            
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    rel_path = str(py_file.relative_to(repo_dir))
                    
                    for line_num, line in enumerate(f, 1):
                        match = import_pattern.match(line)
                        if match:
                            from_module = match.group(1) or ""
                            import_names = [
                                name.strip()
                                for name in match.group(2).split(",")
                            ]
                            
                            relation = ImportRelation(
                                from_file=rel_path,
                                to_module=from_module if from_module else import_names[0],
                                imported_names=import_names,
                                line_number=line_num,
                            )
                            self.imports.append(relation)
                            
            except Exception as e:
                logger.debug(f"Failed to extract imports from {py_file}: {e}")
    
    def _build_call_graph(self, repo_dir: Path) -> None:
        """Build simple call graph (heuristic).
        
        Args:
            repo_dir: Repository root directory
        """
        # Simple heuristic: if a function name appears in a file, assume it's called
        for py_file in repo_dir.rglob("*.py"):
            if any(part in [".git", "__pycache__", "venv", ".venv"]
                   for part in py_file.parts):
                continue
            
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    rel_path = str(py_file.relative_to(repo_dir))
                    
                    # Check for each known symbol
                    for symbol_name in self.symbols.keys():
                        if symbol_name in content:
                            self.call_graph[symbol_name].add(rel_path)
                            
            except Exception as e:
                logger.debug(f"Failed to build call graph for {py_file}: {e}")
    
    def _resolve_module(self, module_name: str) -> Optional[str]:
        """Resolve module name to file path.
        
        Args:
            module_name: Module name (e.g., "foo.bar.baz")
            
        Returns:
            File path if found
        """
        # Simple heuristic: convert dots to slashes and add .py
        potential_path = module_name.replace(".", "/") + ".py"
        
        # Check if any indexed file matches
        for symbol_list in self.symbols.values():
            for symbol in symbol_list:
                if symbol.file_path == potential_path:
                    return potential_path
        
        return None


def build_symbol_index(repo_dir: Path) -> SymbolIndex:
    """Build symbol index for repository.
    
    Args:
        repo_dir: Repository root directory
        
    Returns:
        Symbol index
    """
    index = SymbolIndex()
    index.build(repo_dir)
    return index
