"""Patch application module.

Safely applies patches with rollback support.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .types import Patch, PatchStatus
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ApplyResult:
    """Result of applying a patch."""
    
    success: bool
    patch_id: str
    message: str
    stdout: str = ""
    stderr: str = ""
    files_modified: list = None
    backup_dir: Optional[Path] = None
    
    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []


class PatchApplicator:
    """Apply patches safely with rollback support."""
    
    def __init__(self, repo_dir: Path):
        """Initialize patch applicator.
        
        Args:
            repo_dir: Repository directory
        """
        self.repo_dir = Path(repo_dir)
        self.backup_dir = self.repo_dir / ".rfsn_backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def apply_patch(
        self,
        patch: Patch,
        dry_run: bool = False,
        create_backup: bool = True,
    ) -> ApplyResult:
        """Apply a patch to the repository.
        
        Args:
            patch: Patch to apply
            dry_run: If True, only test application without modifying files
            create_backup: If True, create backup before applying
            
        Returns:
            ApplyResult with status and details
        """
        logger.info(f"Applying patch {patch.patch_id} (dry_run={dry_run})")
        
        # Create backup if requested
        backup_path = None
        if create_backup and not dry_run:
            backup_path = self._create_backup(patch)
        
        # Try to apply patch
        try:
            if patch.file_diffs:
                # Apply file-by-file
                result = self._apply_file_diffs(patch, dry_run)
            else:
                # Apply unified diff
                result = self._apply_unified_diff(patch, dry_run)
            
            if result.success and not dry_run:
                patch.status = PatchStatus.APPLIED
                logger.info(f"Successfully applied patch {patch.patch_id}")
            
            result.backup_dir = backup_path
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply patch {patch.patch_id}: {e}")
            
            # Rollback if backup exists
            if backup_path and not dry_run:
                self.rollback(backup_path)
            
            return ApplyResult(
                success=False,
                patch_id=patch.patch_id,
                message=f"Application failed: {e}",
                stderr=str(e),
            )
    
    def rollback(self, backup_path: Path) -> bool:
        """Rollback to a backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            True if rollback successful
        """
        logger.info(f"Rolling back to backup: {backup_path}")
        
        try:
            # Restore files from backup
            for backup_file in backup_path.rglob("*"):
                if backup_file.is_file():
                    rel_path = backup_file.relative_to(backup_path)
                    target_file = self.repo_dir / rel_path
                    
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_file)
            
            logger.info("Rollback successful")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def verify_applied(self, patch: Patch) -> bool:
        """Verify that a patch was correctly applied.
        
        Args:
            patch: Patch to verify
            
        Returns:
            True if patch is correctly applied
        """
        try:
            for file_diff in patch.file_diffs:
                target_file = self.repo_dir / file_diff.file_path
                
                if not target_file.exists():
                    logger.warning(f"File not found after patch: {target_file}")
                    return False
                
                with open(target_file, 'r', encoding='utf-8') as f:
                    current_content = f.read()
                
                if current_content != file_diff.new_content:
                    logger.warning(f"File content mismatch: {target_file}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def _create_backup(self, patch: Patch) -> Path:
        """Create backup of files that will be modified.
        
        Args:
            patch: Patch to create backup for
            
        Returns:
            Path to backup directory
        """
        backup_path = self.backup_dir / patch.patch_id
        backup_path.mkdir(exist_ok=True)
        
        for file_diff in patch.file_diffs:
            source_file = self.repo_dir / file_diff.file_path
            
            if source_file.exists():
                target_file = backup_path / file_diff.file_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
        
        logger.debug(f"Created backup at {backup_path}")
        return backup_path
    
    def _apply_file_diffs(self, patch: Patch, dry_run: bool) -> ApplyResult:
        """Apply patch using file diffs.
        
        Args:
            patch: Patch to apply
            dry_run: If True, only test application
            
        Returns:
            ApplyResult
        """
        modified_files = []
        
        for file_diff in patch.file_diffs:
            target_file = self.repo_dir / file_diff.file_path
            
            # Check if file exists and matches old content
            if target_file.exists() and file_diff.old_content:
                with open(target_file, 'r', encoding='utf-8') as f:
                    current_content = f.read()
                
                if current_content != file_diff.old_content:
                    return ApplyResult(
                        success=False,
                        patch_id=patch.patch_id,
                        message=f"File {file_diff.file_path} has been modified",
                    )
            
            # Apply changes
            if not dry_run:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                if file_diff.new_content:
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(file_diff.new_content)
                    modified_files.append(str(file_diff.file_path))
                elif target_file.exists():
                    # Delete file
                    target_file.unlink()
                    modified_files.append(str(file_diff.file_path))
        
        return ApplyResult(
            success=True,
            patch_id=patch.patch_id,
            message=f"Applied {len(modified_files)} file changes",
            files_modified=modified_files,
        )
    
    def _apply_unified_diff(self, patch: Patch, dry_run: bool) -> ApplyResult:
        """Apply patch using git apply or patch command.
        
        Args:
            patch: Patch to apply
            dry_run: If True, only test application
            
        Returns:
            ApplyResult
        """
        # Write patch to temporary file
        patch_file = self.backup_dir / f"{patch.patch_id}.patch"
        with open(patch_file, 'w', encoding='utf-8') as f:
            f.write(patch.diff_text)
        
        try:
            # Try git apply first
            cmd = ['git', 'apply']
            if dry_run:
                cmd.append('--check')
            cmd.append(str(patch_file))
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                return ApplyResult(
                    success=True,
                    patch_id=patch.patch_id,
                    message="Applied using git apply",
                    stdout=result.stdout,
                )
            else:
                # Try patch command as fallback
                return self._apply_with_patch_command(patch_file, patch.patch_id, dry_run)
                
        except subprocess.TimeoutExpired:
            return ApplyResult(
                success=False,
                patch_id=patch.patch_id,
                message="Application timed out",
            )
        except Exception as e:
            return ApplyResult(
                success=False,
                patch_id=patch.patch_id,
                message=f"Application failed: {e}",
            )
        finally:
            # Clean up patch file
            if patch_file.exists():
                patch_file.unlink()
    
    def _apply_with_patch_command(
        self,
        patch_file: Path,
        patch_id: str,
        dry_run: bool,
    ) -> ApplyResult:
        """Apply patch using the patch command.
        
        Args:
            patch_file: Path to patch file
            patch_id: Patch ID
            dry_run: If True, only test application
            
        Returns:
            ApplyResult
        """
        cmd = ['patch', '-p1']
        if dry_run:
            cmd.append('--dry-run')
        cmd.extend(['-i', str(patch_file)])
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            return ApplyResult(
                success=True,
                patch_id=patch_id,
                message="Applied using patch command",
                stdout=result.stdout,
            )
        else:
            return ApplyResult(
                success=False,
                patch_id=patch_id,
                message="Patch command failed",
                stderr=result.stderr,
            )


def apply_patch_safe(
    patch: Patch,
    repo_dir: Path,
    verify: bool = True,
) -> ApplyResult:
    """Safely apply a patch with automatic rollback on failure.
    
    Args:
        patch: Patch to apply
        repo_dir: Repository directory
        verify: If True, verify application
        
    Returns:
        ApplyResult
    """
    applicator = PatchApplicator(repo_dir)
    
    # Dry run first
    dry_result = applicator.apply_patch(patch, dry_run=True, create_backup=False)
    if not dry_result.success:
        logger.warning(f"Dry run failed for patch {patch.patch_id}")
        return dry_result
    
    # Apply for real
    result = applicator.apply_patch(patch, dry_run=False, create_backup=True)
    
    # Verify if requested
    if verify and result.success:
        if not applicator.verify_applied(patch):
            logger.error("Verification failed, rolling back")
            if result.backup_dir:
                applicator.rollback(result.backup_dir)
            result.success = False
            result.message = "Verification failed, rolled back"
    
    return result
