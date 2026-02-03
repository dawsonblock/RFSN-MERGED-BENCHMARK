"""AST-based fault localization for precise code targeting.

Uses Python's AST module to analyze code structure and precisely
identify functions, classes, and scopes affected by errors.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeLocation:
    """Precise location of a code element."""
    file: str
    start_line: int
    end_line: int
    name: str
    kind: str  # "function", "class", "method", "module"
    parent: str | None = None  # Parent class/function name
    signature: str = ""
    docstring: str = ""
    
    def contains_line(self, line: int) -> bool:
        """Check if this location contains the given line."""
        return self.start_line <= line <= self.end_line
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "name": self.name,
            "kind": self.kind,
            "parent": self.parent,
            "signature": self.signature,
        }


@dataclass
class FaultLocation:
    """A located fault with context."""
    code_location: CodeLocation
    error_line: int
    confidence: float  # 0.0 - 1.0
    reason: str
    context_lines: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.code_location.to_dict(),
            "error_line": self.error_line,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class ASTAnalyzer:
    """
    AST-based code analyzer for fault localization.
    
    Capabilities:
    - Parse Python files into AST
    - Extract function/class/method locations
    - Find enclosing scope for any line
    - Track variable definitions and usages
    - Identify import dependencies
    """
    
    def __init__(self, source: str, filename: str = "<unknown>"):
        self.source = source
        self.filename = filename
        self.lines = source.split("\n")
        self._tree: ast.AST | None = None
        self._locations: list[CodeLocation] = []
        self._imports: dict[str, str] = {}  # name -> module
        self._definitions: dict[str, int] = {}  # name -> line
        
    def parse(self) -> bool:
        """Parse the source into AST."""
        try:
            self._tree = ast.parse(self.source, filename=self.filename)
            self._extract_locations()
            self._extract_imports()
            return True
        except SyntaxError as e:
            logger.warning("Failed to parse %s: %s", self.filename, e)
            return False
    
    def _extract_locations(self) -> None:
        """Extract all code locations from AST."""
        if not self._tree:
            return
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self._add_function_location(node, parent=None)
            elif isinstance(node, ast.ClassDef):
                self._add_class_location(node)
    
    def _add_function_location(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent: str | None,
    ) -> None:
        """Add a function/method location."""
        # Get signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"def {node.name}({', '.join(args)})"
        
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Determine end line
        end_line = self._get_end_line(node)
        
        kind = "method" if parent else "function"
        
        self._locations.append(CodeLocation(
            file=self.filename,
            start_line=node.lineno,
            end_line=end_line,
            name=node.name,
            kind=kind,
            parent=parent,
            signature=signature,
            docstring=docstring[:200] if docstring else "",
        ))
        
        # Track definition
        self._definitions[node.name] = node.lineno
    
    def _add_class_location(self, node: ast.ClassDef) -> None:
        """Add a class location and its methods."""
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Get base classes
        bases = [self._get_name(b) for b in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        # Determine end line
        end_line = self._get_end_line(node)
        
        self._locations.append(CodeLocation(
            file=self.filename,
            start_line=node.lineno,
            end_line=end_line,
            name=node.name,
            kind="class",
            parent=None,
            signature=signature,
            docstring=docstring[:200] if docstring else "",
        ))
        
        # Track definition
        self._definitions[node.name] = node.lineno
        
        # Add methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                self._add_function_location(item, parent=node.name)
    
    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line of a node."""
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno
        
        # Fallback: find max line in subtree
        max_line = getattr(node, "lineno", 1)
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                max_line = max(max_line, child.lineno)
        return max_line
    
    def _get_name(self, node: ast.AST) -> str:
        """Get the name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "?"
    
    def _extract_imports(self) -> None:
        """Extract import information."""
        if not self._tree:
            return
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    self._imports[name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    self._imports[name] = f"{module}.{alias.name}"
    
    def find_enclosing_scope(self, line: int) -> CodeLocation | None:
        """Find the smallest scope containing the given line."""
        candidates = [
            loc for loc in self._locations
            if loc.contains_line(line)
        ]
        
        if not candidates:
            return None
        
        # Return the smallest (most specific) scope
        return min(candidates, key=lambda loc: loc.end_line - loc.start_line)
    
    def find_function_at_line(self, line: int) -> CodeLocation | None:
        """Find the function/method at the given line."""
        scope = self.find_enclosing_scope(line)
        if scope and scope.kind in ("function", "method"):
            return scope
        return None
    
    def find_class_at_line(self, line: int) -> CodeLocation | None:
        """Find the class containing the given line."""
        for loc in self._locations:
            if loc.kind == "class" and loc.contains_line(line):
                return loc
        return None
    
    def get_context(self, line: int, window: int = 10) -> list[str]:
        """Get context lines around the given line."""
        start = max(0, line - window - 1)
        end = min(len(self.lines), line + window)
        return self.lines[start:end]
    
    def find_definitions_in_scope(self, scope: CodeLocation) -> list[tuple[str, int]]:
        """Find all variable definitions in a scope."""
        definitions = []
        
        # Find all assignments in the scope's line range
        for node in ast.walk(self._tree):
            if not hasattr(node, "lineno"):
                continue
            if not scope.contains_line(node.lineno):
                continue
            
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions.append((target.id, node.lineno))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                definitions.append((node.id, node.lineno))
        
        return definitions
    
    def find_usages(self, name: str) -> list[int]:
        """Find all usages of a name."""
        usages = []
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Name) and node.id == name:
                usages.append(node.lineno)
        
        return usages
    
    def get_all_locations(self) -> list[CodeLocation]:
        """Get all extracted code locations."""
        return self._locations.copy()
    
    def get_imports(self) -> dict[str, str]:
        """Get all imports: name -> full module path."""
        return self._imports.copy()


class ASTFaultLocalizer:
    """
    AST-based fault localization.
    
    Uses error messages and tracebacks to precisely identify
    the code locations most likely to need modification.
    """
    
    def __init__(self):
        self._analyzers: dict[str, ASTAnalyzer] = {}
    
    def add_file(self, path: str, source: str) -> bool:
        """Add a file to analyze."""
        analyzer = ASTAnalyzer(source, path)
        if analyzer.parse():
            self._analyzers[path] = analyzer
            return True
        return False
    
    def localize_from_traceback(
        self,
        traceback: str,
        k: int = 5,
    ) -> list[FaultLocation]:
        """
        Localize faults from a traceback.
        
        Parses traceback to extract file:line pairs, then uses
        AST analysis to identify precise code locations.
        """
        # Parse traceback for file:line pairs
        locations = self._parse_traceback(traceback)
        
        # Score and rank locations
        faults = []
        for i, (file, line, _) in enumerate(locations):
            if file not in self._analyzers:
                continue
            
            analyzer = self._analyzers[file]
            scope = analyzer.find_enclosing_scope(line)
            
            if scope:
                # Higher confidence for later frames (closer to error)
                base_confidence = 0.5 + (i / len(locations)) * 0.4
                
                # Boost for functions/methods (more actionable)
                if scope.kind in ("function", "method"):
                    base_confidence += 0.1
                
                faults.append(FaultLocation(
                    code_location=scope,
                    error_line=line,
                    confidence=min(base_confidence, 1.0),
                    reason=f"Traceback frame {i+1}",
                    context_lines=analyzer.get_context(line),
                ))
        
        # Sort by confidence descending
        faults.sort(key=lambda f: f.confidence, reverse=True)
        return faults[:k]
    
    def localize_from_error(
        self,
        error_type: str,
        error_message: str,
        file: str,
        line: int | None = None,
    ) -> list[FaultLocation]:
        """
        Localize faults from an error type and message.
        
        Uses pattern matching to identify likely causes.
        """
        faults = []
        
        if file not in self._analyzers:
            return faults
        
        analyzer = self._analyzers[file]
        
        # Error-specific localization
        if error_type == "AttributeError":
            faults.extend(self._localize_attribute_error(
                analyzer, error_message, line
            ))
        elif error_type == "NameError":
            faults.extend(self._localize_name_error(
                analyzer, error_message, line
            ))
        elif error_type == "TypeError":
            faults.extend(self._localize_type_error(
                analyzer, error_message, line
            ))
        elif error_type == "ImportError":
            faults.extend(self._localize_import_error(
                analyzer, error_message
            ))
        
        # Generic: if we have a line, find its scope
        if line and not faults:
            scope = analyzer.find_enclosing_scope(line)
            if scope:
                faults.append(FaultLocation(
                    code_location=scope,
                    error_line=line,
                    confidence=0.6,
                    reason=f"{error_type} at line {line}",
                    context_lines=analyzer.get_context(line),
                ))
        
        return faults
    
    def _parse_traceback(
        self,
        traceback: str,
    ) -> list[tuple[str, int, str]]:
        """Parse traceback for file:line:context tuples."""
        # Pattern: File "path", line N, in <name>
        pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(pattern, traceback)
        
        results = []
        for file_path, line_str in matches:
            line = int(line_str)
            # Extract context line if available
            context = ""
            results.append((file_path, line, context))
        
        return results
    
    def _localize_attribute_error(
        self,
        analyzer: ASTAnalyzer,
        message: str,
        line: int | None,
    ) -> list[FaultLocation]:
        """Localize AttributeError."""
        faults = []
        
        # Extract attribute name from message
        # "object has no attribute 'foo'"
        attr_match = re.search(r"no attribute '(\w+)'", message)
        if attr_match:
            attr_name = attr_match.group(1)
            
            # Find usages of this attribute
            # This would require more sophisticated AST walking
            # For now, use the line if available
            if line:
                scope = analyzer.find_enclosing_scope(line)
                if scope:
                    faults.append(FaultLocation(
                        code_location=scope,
                        error_line=line,
                        confidence=0.8,
                        reason=f"Missing attribute '{attr_name}'",
                        context_lines=analyzer.get_context(line),
                    ))
        
        return faults
    
    def _localize_name_error(
        self,
        analyzer: ASTAnalyzer,
        message: str,
        line: int | None,
    ) -> list[FaultLocation]:
        """Localize NameError."""
        faults = []
        
        # Extract undefined name
        # "name 'foo' is not defined"
        name_match = re.search(r"name '(\w+)' is not defined", message)
        if name_match:
            undefined_name = name_match.group(1)
            
            # Check if it's an import issue
            imports = analyzer.get_imports()
            if undefined_name not in imports and line:
                # Possibly missing import
                scope = analyzer.find_enclosing_scope(line)
                if scope:
                    faults.append(FaultLocation(
                        code_location=scope,
                        error_line=line,
                        confidence=0.85,
                        reason=f"Undefined name '{undefined_name}' - possibly missing import",
                        context_lines=analyzer.get_context(line),
                        dependencies=[undefined_name],
                    ))
        
        return faults
    
    def _localize_type_error(
        self,
        analyzer: ASTAnalyzer,
        message: str,
        line: int | None,
    ) -> list[FaultLocation]:
        """Localize TypeError."""
        faults = []
        
        if line:
            scope = analyzer.find_function_at_line(line)
            if scope:
                faults.append(FaultLocation(
                    code_location=scope,
                    error_line=line,
                    confidence=0.7,
                    reason=f"Type mismatch in {scope.name}",
                    context_lines=analyzer.get_context(line),
                ))
        
        return faults
    
    def _localize_import_error(
        self,
        analyzer: ASTAnalyzer,
        message: str,
    ) -> list[FaultLocation]:
        """Localize ImportError."""
        faults = []
        
        # Find the import statement
        # "cannot import name 'foo' from 'module'"
        import_match = re.search(r"cannot import name '(\w+)'", message)
        if import_match:
            name = import_match.group(1)
            
            # Find where this is imported
            usages = analyzer.find_usages(name)
            if usages:
                line = usages[0]
                faults.append(FaultLocation(
                    code_location=CodeLocation(
                        file=analyzer.filename,
                        start_line=line,
                        end_line=line,
                        name=name,
                        kind="import",
                    ),
                    error_line=line,
                    confidence=0.9,
                    reason=f"Failed import of '{name}'",
                    context_lines=analyzer.get_context(line),
                ))
        
        return faults
    
    def get_function_dependencies(
        self,
        file: str,
        function_name: str,
    ) -> list[str]:
        """Get dependencies of a function."""
        if file not in self._analyzers:
            return []
        
        analyzer = self._analyzers[file]
        
        # Find the function
        for loc in analyzer.get_all_locations():
            if loc.name == function_name and loc.kind in ("function", "method"):
                # Get all names used in the function
                definitions = analyzer.find_definitions_in_scope(loc)
                imports = analyzer.get_imports()
                
                # Return external dependencies (imports used in function)
                deps = []
                for name, _ in definitions:
                    if name in imports:
                        deps.append(imports[name])
                return deps
        
        return []


def localize_faults(
    files: dict[str, str],
    error_output: str,
    k: int = 5,
) -> list[FaultLocation]:
    """
    Convenience function to localize faults from error output.
    
    Args:
        files: Dict of filename -> source code
        error_output: Error/traceback output
        k: Max faults to return
        
    Returns:
        List of FaultLocations sorted by confidence
    """
    localizer = ASTFaultLocalizer()
    
    # Add files
    for path, source in files.items():
        localizer.add_file(path, source)
    
    # Localize from traceback
    faults = localizer.localize_from_traceback(error_output, k=k)
    
    return faults


def analyze_file(path: str | Path) -> ASTAnalyzer | None:
    """
    Analyze a single Python file.
    
    Returns analyzer on success, None on failure.
    """
    path = Path(path)
    if not path.exists() or path.suffix != ".py":
        return None
    
    try:
        source = path.read_text()
        analyzer = ASTAnalyzer(source, str(path))
        if analyzer.parse():
            return analyzer
    except Exception as e:
        logger.warning("Failed to analyze %s: %s", path, e)
    
    return None
