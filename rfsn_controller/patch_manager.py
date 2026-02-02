"""
PatchManager: Extracted patch application logic from controller.

This module handles the lifecycle of patch operations including:
- Patch generation and validation
- Isolated testing in git worktrees
- Diff minimization
- Success/failure tracking
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .diff_minimizer import DiffMinimizer
from .metrics import track_patch_application
from .structured_logging import get_logger
from .workspace_manager import GitResult, WorkspaceManager

logger = get_logger(__name__)


@dataclass
class PatchResult:
    """Result of applying and testing a patch."""
    
    patch_id: str
    content: str
    applied: bool
    tests_passed: bool
    test_output: str
    diff: str
    error: str | None = None


class PatchManager:
    """
    Manages patch application and verification workflow.
    
    The PatchManager is responsible for:
    1. Applying patches in isolated worktrees
    2. Running tests to verify patches
    3. Minimizing diffs for clean patches
    4. Tracking patch success/failure metrics
    
    Example:
        >>> manager = PatchManager(workspace_manager, diff_minimizer)
        >>> result = await manager.apply_and_test(patch_content, test_command)
        >>> if result.tests_passed:
        ...     print(f"Patch {result.patch_id} successful!")
    """
    
    def __init__(
        self,
        workspace: WorkspaceManager,
        diff_minimizer: DiffMinimizer | None = None
    ):
        """
        Initialize the PatchManager.
        
        Args:
            workspace: WorkspaceManager for git operations
            diff_minimizer: Optional diff minimizer for clean patches
        """
        self.workspace = workspace
        self.diff_minimizer = diff_minimizer or DiffMinimizer()
        self.patch_counter = 0
        
    def generate_patch_id(self, content: str) -> str:
        """
        Generate a unique patch ID from content.
        
        Args:
            content: Patch content
            
        Returns:
            Unique patch identifier
        """
        self.patch_counter += 1
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"patch-{self.patch_counter}-{content_hash}"
    
    async def apply_and_test(
        self,
        patch_content: str,
        test_command: list[str],
        base_branch: str = "HEAD"
    ) -> PatchResult:
        """
        Apply patch in isolated worktree and run tests.
        
        Args:
            patch_content: The patch to apply
            test_command: Command to run tests
            base_branch: Base branch for worktree
            
        Returns:
            PatchResult with application and test outcomes
        """
        patch_id = self.generate_patch_id(patch_content)
        
        logger.info(
            "Applying and testing patch",
            patch_id=patch_id,
            test_command=" ".join(test_command)
        )
        
        with track_patch_application(phase="verification"):
            try:
                # Create isolated worktree
                worktree_path = self.workspace.make_worktree(
                    branch=f"patch-{patch_id}",
                    base_branch=base_branch
                )
                
                try:
                    # Apply patch
                    apply_result = await self._apply_patch(
                        worktree_path,
                        patch_content
                    )
                    
                    if not apply_result.success:
                        return PatchResult(
                            patch_id=patch_id,
                            content=patch_content,
                            applied=False,
                            tests_passed=False,
                            test_output="",
                            diff="",
                            error=f"Failed to apply patch: {apply_result.stderr}"
                        )
                    
                    # Run tests
                    test_result = await self._run_tests(
                        worktree_path,
                        test_command
                    )
                    
                    # Get diff
                    diff = self.workspace.get_diff()
                    
                    # Minimize diff if requested
                    if self.diff_minimizer:
                        diff = self.diff_minimizer.minimize(diff)
                    
                    return PatchResult(
                        patch_id=patch_id,
                        content=patch_content,
                        applied=True,
                        tests_passed=test_result.success,
                        test_output=test_result.output,
                        diff=diff,
                        error=None if test_result.success else test_result.stderr
                    )
                    
                finally:
                    # Cleanup worktree
                    self.workspace.cleanup_worktree(worktree_path)
                    
            except Exception as e:
                logger.error("Patch application failed", patch_id=patch_id, error=str(e))
                return PatchResult(
                    patch_id=patch_id,
                    content=patch_content,
                    applied=False,
                    tests_passed=False,
                    test_output="",
                    diff="",
                    error=str(e)
                )
    
    async def _apply_patch(self, worktree_path: str, patch_content: str) -> GitResult:
        """Apply patch content to worktree."""
        # Write patch to file
        patch_file = Path(worktree_path) / ".patch"
        patch_file.write_text(patch_content)
        
        # Apply using git
        result = self.workspace.run_git(
            ["apply", str(patch_file)],
            cwd=worktree_path
        )
        
        # Clean up patch file
        patch_file.unlink()
        
        return result
    
    async def _run_tests(self, worktree_path: str, test_command: list[str]) -> GitResult:
        """Run test command in worktree."""
        import subprocess
        
        try:
            result = subprocess.run(
                test_command,
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=300, check=False  # 5 minute timeout
            )
            
            return GitResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0
            )
            
        except subprocess.TimeoutExpired:
            return GitResult(
                returncode=-1,
                stdout="",
                stderr="Test command timed out",
                success=False
            )
        except Exception as e:
            return GitResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
                success=False
            )
