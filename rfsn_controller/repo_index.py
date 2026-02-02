"""Repository index for structural understanding.

This module provides a lightweight index of the repository structure including:
- File list with metadata
- Language detection
- Python import graph
- Symbol map (functions/classes)

The index is built without network access and persisted for reuse.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Language detection by extension
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
}

# Directories to skip when indexing
SKIP_DIRS: set[str] = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "egg-info",
    ".eggs",
}


@dataclass
class FileInfo:
    """Metadata about a file in the repository."""
    
    path: str
    size: int
    mtime: float
    language: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime": self.mtime,
            "language": self.language,
        }


@dataclass
class SymbolInfo:
    """Information about a code symbol (function, class, etc.)."""
    
    name: str
    kind: str  # "function", "class", "method"
    file: str
    line: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
        }


@dataclass
class RepoIndex:
    """Index of repository structure and symbols.
    
    Features:
    - File list with sizes and languages
    - Python import dependency graph
    - Symbol map for Python files
    """
    
    root: str
    files: list[FileInfo] = field(default_factory=list)
    import_graph: dict[str, set[str]] = field(default_factory=dict)
    symbols: list[SymbolInfo] = field(default_factory=list)
    _hash: str = ""
    
    @classmethod
    def build(cls, root: str, max_files: int = 10000) -> RepoIndex:
        """Build a new index from a repository root.
        
        Args:
            root: Path to repository root.
            max_files: Maximum number of files to index.
            
        Returns:
            Populated RepoIndex.
        """
        root_path = Path(root).resolve()
        index = cls(root=str(root_path))
        
        # Walk the directory tree
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            
            for filename in filenames:
                if file_count >= max_files:
                    break
                
                filepath = Path(dirpath) / filename
                try:
                    stat = filepath.stat()
                    rel_path = str(filepath.relative_to(root_path))
                    
                    # Detect language
                    ext = filepath.suffix.lower()
                    language = LANGUAGE_MAP.get(ext)
                    
                    file_info = FileInfo(
                        path=rel_path,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        language=language,
                    )
                    index.files.append(file_info)
                    file_count += 1
                    
                    # Extract Python imports and symbols
                    if language == "python" and stat.st_size < 500_000:
                        index._extract_python_info(filepath, rel_path)
                        
                except (OSError, PermissionError):
                    continue
        
        # Sort files for deterministic ordering
        index.files.sort(key=lambda f: f.path)
        index.symbols.sort(key=lambda s: (s.file, s.line))
        
        # Compute hash
        index._hash = index._compute_hash()
        
        return index
    
    def _extract_python_info(self, filepath: Path, rel_path: str) -> None:
        """Extract imports and symbols from a Python file.
        
        Args:
            filepath: Absolute path to the file.
            rel_path: Relative path for indexing.
        """
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return
        
        imports: set[str] = set()
        
        for node in ast.walk(tree):
            # Collect imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
            
            # Collect top-level functions and classes
            elif isinstance(node, ast.FunctionDef) and self._is_top_level(node, tree) or isinstance(node, ast.AsyncFunctionDef) and self._is_top_level(node, tree):
                self.symbols.append(SymbolInfo(
                    name=node.name,
                    kind="function",
                    file=rel_path,
                    line=node.lineno,
                ))
            elif isinstance(node, ast.ClassDef) and self._is_top_level(node, tree):
                self.symbols.append(SymbolInfo(
                    name=node.name,
                    kind="class",
                    file=rel_path,
                    line=node.lineno,
                ))
        
        if imports:
            self.import_graph[rel_path] = imports
    
    def _is_top_level(self, node: ast.AST, tree: ast.Module) -> bool:
        """Check if a node is at the top level of the module."""
        return node in tree.body
    
    def _compute_hash(self) -> str:
        """Compute a deterministic hash of the index content."""
        content = json.dumps(self.to_json(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_import_graph(self) -> dict[str, set[str]]:
        """Get the import dependency graph.
        
        Returns:
            Dict mapping file paths to sets of imported module names.
        """
        return self.import_graph.copy()
    
    def search_symbols(self, name: str) -> list[SymbolInfo]:
        """Search for symbols by name substring.
        
        Args:
            name: Name substring to search for (case-insensitive).
            
        Returns:
            List of matching symbols.
        """
        name_lower = name.lower()
        return [s for s in self.symbols if name_lower in s.name.lower()]
    
    def get_files_by_language(self, language: str) -> list[FileInfo]:
        """Get all files of a specific language.
        
        Args:
            language: Language name (e.g., "python").
            
        Returns:
            List of matching files.
        """
        return [f for f in self.files if f.language == language]
    
    def to_json(self) -> dict[str, Any]:
        """Convert index to JSON-serializable dict.
        
        Returns:
            Dictionary representation of the index.
        """
        return {
            "root": self.root,
            "file_count": len(self.files),
            "symbol_count": len(self.symbols),
            "hash": self._hash,
            "files": [f.to_dict() for f in self.files],
            "import_graph": {k: list(v) for k, v in self.import_graph.items()},
            "symbols": [s.to_dict() for s in self.symbols],
        }
    
    def to_compact_json(self, max_files: int = 50, max_symbols: int = 100) -> dict[str, Any]:
        """Get a compact representation for LLM context.
        
        Args:
            max_files: Maximum files to include.
            max_symbols: Maximum symbols to include.
            
        Returns:
            Compact dictionary for prompts.
        """
        return {
            "file_count": len(self.files),
            "languages": list({f.language for f in self.files if f.language}),
            "top_files": [f.path for f in self.files[:max_files]],
            "symbols": [f"{s.kind}:{s.name}" for s in self.symbols[:max_symbols]],
        }
    
    def save(self, path: str) -> None:
        """Save the index to a JSON file.
        
        Args:
            path: Path to save the index.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> RepoIndex:
        """Load an index from a JSON file.
        
        Args:
            path: Path to the saved index.
            
        Returns:
            Loaded RepoIndex.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        index = cls(root=data["root"])
        index._hash = data.get("hash", "")
        
        for f_data in data.get("files", []):
            index.files.append(FileInfo(**f_data))
        
        for path_key, imports in data.get("import_graph", {}).items():
            index.import_graph[path_key] = set(imports)
        
        for s_data in data.get("symbols", []):
            index.symbols.append(SymbolInfo(**s_data))
        
        return index
