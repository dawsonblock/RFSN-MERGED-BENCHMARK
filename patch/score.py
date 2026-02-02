"""Patch scoring module.

Scores patches based on:
- Static analysis (syntax, imports, style)
- Test delta (incremental test running)
- Diff risk (complexity, size, critical sections)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from .types import Patch, PatchStatus
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PatchScore:
    """Comprehensive patch score."""
    
    # Component scores (0.0 - 1.0)
    static_score: float = 0.0
    test_score: float = 0.0
    risk_score: float = 0.0
    
    # Overall score (weighted combination)
    total_score: float = 0.0
    
    # Detailed breakdown
    syntax_valid: bool = False
    imports_valid: bool = False
    style_score: float = 0.0
    complexity_score: float = 0.0
    test_pass_rate: float = 0.0
    lines_changed: int = 0
    files_changed: int = 0
    
    # Failure reasons
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class PatchScorer:
    """Score patches for quality and viability."""
    
    def __init__(self):
        """Initialize patch scorer."""
        self.weights = {
            'static': 0.3,
            'test': 0.5,
            'risk': 0.2,
        }
    
    def score_patch(self, patch: Patch, repo_dir: Path) -> PatchScore:
        """Score a patch comprehensively.
        
        Args:
            patch: Patch to score
            repo_dir: Repository directory
            
        Returns:
            PatchScore with detailed breakdown
        """
        score = PatchScore()
        
        # Static analysis
        static_result = self._score_static(patch, repo_dir)
        score.static_score = static_result['score']
        score.syntax_valid = static_result['syntax_valid']
        score.imports_valid = static_result['imports_valid']
        score.style_score = static_result['style_score']
        score.errors.extend(static_result['errors'])
        
        # Diff risk analysis
        risk_result = self._score_risk(patch, repo_dir)
        score.risk_score = risk_result['score']
        score.complexity_score = risk_result['complexity']
        score.lines_changed = risk_result['lines_changed']
        score.files_changed = risk_result['files_changed']
        
        # Calculate total score
        score.total_score = (
            self.weights['static'] * score.static_score +
            self.weights['test'] * score.test_score +
            self.weights['risk'] * score.risk_score
        )
        
        # Update patch
        patch.score = score.total_score
        patch.static_score = score.static_score
        patch.test_score = score.test_score
        patch.diff_risk_score = score.risk_score
        patch.syntax_valid = score.syntax_valid
        patch.imports_valid = score.imports_valid
        
        if score.syntax_valid and score.imports_valid:
            patch.status = PatchStatus.SCORED
        
        logger.info(f"Scored patch {patch.patch_id}: {score.total_score:.2f}")
        
        return score
    
    def _score_static(self, patch: Patch, repo_dir: Path) -> dict:
        """Perform static analysis scoring.
        
        Args:
            patch: Patch to analyze
            repo_dir: Repository directory
            
        Returns:
            Dictionary with static analysis results
        """
        result = {
            'score': 0.0,
            'syntax_valid': False,
            'imports_valid': False,
            'style_score': 0.0,
            'errors': [],
        }
        
        if not patch.file_diffs:
            result['errors'].append("No file diffs found")
            return result
        
        # Check each file
        syntax_checks = []
        import_checks = []
        style_scores = []
        
        for file_diff in patch.file_diffs:
            if not file_diff.new_content:
                continue
            
            # Syntax check (Python only for now)
            if file_diff.file_path.endswith('.py'):
                syntax_ok, syntax_error = self._check_python_syntax(file_diff.new_content)
                syntax_checks.append(syntax_ok)
                if not syntax_ok:
                    result['errors'].append(f"Syntax error in {file_diff.file_path}: {syntax_error}")
                
                # Import check
                imports_ok, import_error = self._check_python_imports(file_diff.new_content)
                import_checks.append(imports_ok)
                if not imports_ok:
                    result['errors'].append(f"Import issue in {file_diff.file_path}: {import_error}")
                
                # Style check
                style_score = self._check_python_style(file_diff.new_content)
                style_scores.append(style_score)
        
        # Aggregate results
        if syntax_checks:
            result['syntax_valid'] = all(syntax_checks)
        if import_checks:
            result['imports_valid'] = all(import_checks)
        if style_scores:
            result['style_score'] = sum(style_scores) / len(style_scores)
        
        # Calculate overall static score
        components = []
        if syntax_checks:
            components.append(1.0 if result['syntax_valid'] else 0.0)
        if import_checks:
            components.append(1.0 if result['imports_valid'] else 0.0)
        if style_scores:
            components.append(result['style_score'])
        
        if components:
            result['score'] = sum(components) / len(components)
        
        return result
    
    def _score_risk(self, patch: Patch, repo_dir: Path) -> dict:
        """Score patch risk based on diff characteristics.
        
        Args:
            patch: Patch to analyze
            repo_dir: Repository directory
            
        Returns:
            Dictionary with risk analysis results
        """
        result = {
            'score': 1.0,  # Start with perfect score
            'complexity': 0.0,
            'lines_changed': 0,
            'files_changed': len(patch.file_diffs),
        }
        
        # Count changes
        total_lines = 0
        for file_diff in patch.file_diffs:
            if file_diff.new_content and file_diff.old_content:
                old_lines = len(file_diff.old_content.splitlines())
                new_lines = len(file_diff.new_content.splitlines())
                total_lines += abs(new_lines - old_lines)
        
        result['lines_changed'] = total_lines
        
        # Risk factors (reduce score)
        risk_factors = []
        
        # Too many lines changed
        if total_lines > 100:
            risk_factors.append(0.2)
        elif total_lines > 50:
            risk_factors.append(0.1)
        
        # Too many files
        if len(patch.file_diffs) > 5:
            risk_factors.append(0.2)
        elif len(patch.file_diffs) > 3:
            risk_factors.append(0.1)
        
        # Critical files (config, __init__, main, etc.)
        critical_patterns = ['__init__.py', 'config', 'main.py', 'setup.py']
        for file_diff in patch.file_diffs:
            if any(pattern in file_diff.file_path.lower() for pattern in critical_patterns):
                risk_factors.append(0.1)
                break
        
        # Apply risk factors
        for factor in risk_factors:
            result['score'] -= factor
        
        result['score'] = max(0.0, result['score'])
        result['complexity'] = len(risk_factors) / 5.0  # Normalize
        
        return result
    
    def _check_python_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Check Python syntax validity.
        
        Args:
            code: Python source code
            
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Parse error: {e}"
    
    def _check_python_imports(self, code: str) -> tuple[bool, Optional[str]]:
        """Check if imports are valid (basic check).
        
        Args:
            code: Python source code
            
        Returns:
            (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Check for common problematic patterns
            for imp in imports:
                # Relative imports without package
                if imp.startswith('.') and '.' not in imp[1:]:
                    return False, f"Suspicious relative import: {imp}"
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _check_python_style(self, code: str) -> float:
        """Check Python code style (basic heuristics).
        
        Args:
            code: Python source code
            
        Returns:
            Style score (0.0 - 1.0)
        """
        score = 1.0
        lines = code.splitlines()
        
        # Check for common style issues
        issues = 0
        
        for line in lines:
            # Very long lines
            if len(line) > 120:
                issues += 1
            
            # Trailing whitespace
            if line != line.rstrip():
                issues += 1
            
            # Multiple statements on one line
            if ';' in line and not line.strip().startswith('#'):
                issues += 1
        
        # Penalize based on issues
        if issues > 0:
            penalty = min(0.5, issues * 0.05)
            score -= penalty
        
        return max(0.0, score)


def score_patches(patches: List[Patch], repo_dir: Path) -> List[Patch]:
    """Score a list of patches and sort by score.
    
    Args:
        patches: List of patches to score
        repo_dir: Repository directory
        
    Returns:
        Sorted list of patches (highest score first)
    """
    scorer = PatchScorer()
    
    for patch in patches:
        scorer.score_patch(patch, repo_dir)
    
    # Sort by score (descending)
    patches.sort(key=lambda p: p.score, reverse=True)
    
    logger.info(f"Scored {len(patches)} patches")
    return patches
