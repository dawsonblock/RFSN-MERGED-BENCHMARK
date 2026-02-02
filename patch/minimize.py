"""Patch minimization module.

Minimizes patches by:
- Removing unnecessary changes
- Splitting multi-file patches
- Delta debugging for minimal failing diff
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import List, Optional

from .types import Patch, FileDiff, PatchStatus
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


class PatchMinimizer:
    """Minimize patches to essential changes only."""
    
    def __init__(self):
        """Initialize patch minimizer."""
        pass
    
    def minimize_patch(self, patch: Patch, repo_dir: Path) -> Patch:
        """Minimize a patch to essential changes.
        
        Args:
            patch: Patch to minimize
            repo_dir: Repository directory
            
        Returns:
            Minimized patch
        """
        logger.info(f"Minimizing patch {patch.patch_id}")
        
        minimized_diffs = []
        
        for file_diff in patch.file_diffs:
            minimized_diff = self._minimize_file_diff(file_diff)
            if minimized_diff:
                minimized_diffs.append(minimized_diff)
        
        # Update patch
        patch.file_diffs = minimized_diffs
        patch.diff_text = self._generate_unified_diff(minimized_diffs)
        
        logger.info(f"Minimized patch from {len(patch.file_diffs)} to {len(minimized_diffs)} files")
        
        return patch
    
    def split_patch(self, patch: Patch) -> List[Patch]:
        """Split multi-file patch into single-file patches.
        
        Args:
            patch: Patch to split
            
        Returns:
            List of single-file patches
        """
        if len(patch.file_diffs) <= 1:
            return [patch]
        
        logger.info(f"Splitting patch {patch.patch_id} into {len(patch.file_diffs)} patches")
        
        patches = []
        for i, file_diff in enumerate(patch.file_diffs):
            new_patch = Patch(
                patch_id=f"{patch.patch_id}_split_{i}",
                strategy=patch.strategy,
                diff_text=self._generate_unified_diff([file_diff]),
                file_diffs=[file_diff],
                status=patch.status,
                localization_hits=patch.localization_hits,
                rationale=f"Split from {patch.patch_id}",
                generation_method=patch.generation_method,
                metadata={'parent_id': patch.patch_id, 'split_index': i},
            )
            patches.append(new_patch)
        
        return patches
    
    def remove_whitespace_changes(self, patch: Patch) -> Patch:
        """Remove whitespace-only changes from patch.
        
        Args:
            patch: Patch to clean
            
        Returns:
            Cleaned patch
        """
        cleaned_diffs = []
        
        for file_diff in patch.file_diffs:
            if file_diff.old_content and file_diff.new_content:
                old_stripped = self._normalize_whitespace(file_diff.old_content)
                new_stripped = self._normalize_whitespace(file_diff.new_content)
                
                if old_stripped != new_stripped:
                    # Has non-whitespace changes
                    cleaned_diffs.append(file_diff)
                else:
                    logger.debug(f"Removed whitespace-only changes from {file_diff.file_path}")
            else:
                # New or deleted file
                cleaned_diffs.append(file_diff)
        
        patch.file_diffs = cleaned_diffs
        patch.diff_text = self._generate_unified_diff(cleaned_diffs)
        
        return patch
    
    def delta_debug(
        self,
        patch: Patch,
        test_func: callable,
        granularity: str = 'line'
    ) -> Patch:
        """Use delta debugging to find minimal failing patch.
        
        Args:
            patch: Patch to minimize
            test_func: Function that tests if patch works (returns bool)
            granularity: 'line' or 'hunk' level debugging
            
        Returns:
            Minimal patch that still fails tests
        """
        logger.info(f"Delta debugging patch {patch.patch_id}")
        
        if granularity == 'line':
            return self._delta_debug_lines(patch, test_func)
        else:
            return self._delta_debug_hunks(patch, test_func)
    
    def _minimize_file_diff(self, file_diff: FileDiff) -> Optional[FileDiff]:
        """Minimize changes in a single file diff.
        
        Args:
            file_diff: File diff to minimize
            
        Returns:
            Minimized file diff or None if no changes
        """
        if not file_diff.old_content or not file_diff.new_content:
            return file_diff
        
        old_lines = file_diff.old_content.splitlines(keepends=True)
        new_lines = file_diff.new_content.splitlines(keepends=True)
        
        # Find actual differences
        differ = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        # Check if there are meaningful changes
        if differ.ratio() > 0.99:  # 99% similar
            logger.debug(f"Skipping nearly identical file: {file_diff.file_path}")
            return None
        
        # Keep the diff as is (more sophisticated minimization could be added)
        return file_diff
    
    def _delta_debug_lines(self, patch: Patch, test_func: callable) -> Patch:
        """Delta debug at line level.
        
        Args:
            patch: Patch to debug
            test_func: Test function
            
        Returns:
            Minimal patch
        """
        # Simplified implementation
        # Full implementation would use the delta debugging algorithm
        logger.warning("Line-level delta debugging not fully implemented")
        return patch
    
    def _delta_debug_hunks(self, patch: Patch, test_func: callable) -> Patch:
        """Delta debug at hunk level.
        
        Args:
            patch: Patch to debug
            test_func: Test function
            
        Returns:
            Minimal patch
        """
        # Simplified implementation
        logger.warning("Hunk-level delta debugging not fully implemented")
        return patch
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        lines = text.splitlines()
        normalized = [line.rstrip() for line in lines]
        return '\n'.join(normalized)
    
    def _generate_unified_diff(self, file_diffs: List[FileDiff]) -> str:
        """Generate unified diff from file diffs.
        
        Args:
            file_diffs: List of file diffs
            
        Returns:
            Unified diff text
        """
        diff_parts = []
        
        for file_diff in file_diffs:
            if file_diff.unified_diff:
                diff_parts.append(file_diff.unified_diff)
            elif file_diff.old_content and file_diff.new_content:
                # Generate diff
                old_lines = file_diff.old_content.splitlines(keepends=True)
                new_lines = file_diff.new_content.splitlines(keepends=True)
                
                diff = difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{file_diff.file_path}",
                    tofile=f"b/{file_diff.file_path}",
                )
                diff_parts.append(''.join(diff))
        
        return '\n'.join(diff_parts)


def minimize_patches(patches: List[Patch], repo_dir: Path) -> List[Patch]:
    """Minimize a list of patches.
    
    Args:
        patches: List of patches to minimize
        repo_dir: Repository directory
        
    Returns:
        List of minimized patches
    """
    minimizer = PatchMinimizer()
    
    minimized = []
    for patch in patches:
        try:
            minimized_patch = minimizer.minimize_patch(patch, repo_dir)
            minimized_patch = minimizer.remove_whitespace_changes(minimized_patch)
            minimized.append(minimized_patch)
        except Exception as e:
            logger.error(f"Failed to minimize patch {patch.patch_id}: {e}")
            minimized.append(patch)
    
    logger.info(f"Minimized {len(minimized)} patches")
    return minimized
