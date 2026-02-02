"""Semantic Guardrails - Lightweight static analysis for safety.

Enforces constraints like "do not change function signatures" or
"do not modify public API" to prevent silent regressions.
"""

from __future__ import annotations

import ast


class SemanticDiff:
    """Analyzes semantic changes in code."""
    
    def __init__(self, old_code: str, new_code: str):
        self.old_tree = self._parse(old_code)
        self.new_tree = self._parse(new_code)
        
    def _parse(self, code: str) -> ast.AST | None:
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def get_changed_functions(self) -> list[str]:
        """Get list of functions with changed signatures."""
        if not self.old_tree or not self.new_tree:
            return []
            
        old_funcs = self._extract_functions(self.old_tree)
        new_funcs = self._extract_functions(self.new_tree)
        
        changed = []
        for name, sig in old_funcs.items():
            if name in new_funcs:
                if new_funcs[name] != sig:
                    changed.append(name)
        return changed
        
    def _extract_functions(self, tree: ast.AST) -> dict[str, str]:
        """Extract function signatures."""
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Simple signature hash: name + args
                # We normalize to catch only real signature changes (not body)
                args = [a.arg for a in node.args.args]
                funcs[node.name] = f"{node.name}({','.join(args)})"
        return funcs


class SemanticGuardrails:
    """Enforces semantic safety checks."""
    
    def check_diff(self, file_path: str, old_content: str, new_content: str) -> list[str]:
        """Check if a file change violates safety guardrails.
        
        Args:
            file_path: Path to changed file.
            old_content: Original content.
            new_content: New content.
            
        Returns:
            List of violation messages.
        """
        violations = []
        
        # 1. Parse check
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            violations.append(f"Syntax error in {file_path}: {e}")
            return violations
            
        # 2. Signature check
        # For repair mode, we generally want to avoid changing function signatures
        # unless explicitly requested.
        sd = SemanticDiff(old_content, new_content)
        changed_funcs = sd.get_changed_functions()
        if changed_funcs:
            violations.append(
                f"Changed function signatures in {file_path}: {', '.join(changed_funcs)}"
            )
            
        return violations
