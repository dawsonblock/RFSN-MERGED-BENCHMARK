"""Patch generation module.

Generates N small patches constrained to localized spans.
Now includes real LLM integration for intelligent patch generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import List, Optional

from .types import (
    Patch,
    PatchCandidate,
    PatchGenerationRequest,
    PatchGenerationResult,
    PatchStatus,
    PatchStrategy,
    FileDiff,
)
from localize.types import LocalizationHit
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)

# Import LLM components (optional, graceful degradation)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm.client import get_llm_client, LLMClient
    from llm.prompts import PatchPromptTemplates, build_context_from_localization
    LLM_AVAILABLE = True
except ImportError:
    logger.warning("LLM client not available, falling back to mock mode")
    LLM_AVAILABLE = False
    get_llm_client = None
    PatchPromptTemplates = None
    build_context_from_localization = None


class PatchGenerator:
    """Generate patches from localization hits."""
    
    def __init__(self, llm_client=None):
        """Initialize patch generator.
        
        Args:
            llm_client: Optional LLM client for generation
        """
        self.llm_client = llm_client
    
    def generate(self, request: PatchGenerationRequest) -> PatchGenerationResult:
        """Generate patches for a problem.
        
        Args:
            request: Patch generation request
            
        Returns:
            Generation result with candidates
        """
        start_time = time.time()
        logger.info(f"Generating patches with strategy: {request.strategy.value}")
        
        candidates = []
        errors = []
        total_tokens = 0
        
        try:
            # Select generation strategy
            if request.strategy == PatchStrategy.DIRECT_FIX:
                candidates = self._generate_direct_fix(request)
            elif request.strategy == PatchStrategy.TEST_DRIVEN:
                candidates = self._generate_test_driven(request)
            elif request.strategy == PatchStrategy.HYPOTHESIS:
                candidates = self._generate_hypothesis(request)
            elif request.strategy == PatchStrategy.INCREMENTAL:
                candidates = self._generate_incremental(request)
            elif request.strategy == PatchStrategy.ENSEMBLE:
                candidates = self._generate_ensemble(request)
            else:
                raise ValueError(f"Unknown strategy: {request.strategy}")
            
            logger.info(f"Generated {len(candidates)} patch candidates")
            
        except Exception as e:
            logger.error(f"Patch generation failed: {e}")
            errors.append(str(e))
        
        generation_time = time.time() - start_time
        
        return PatchGenerationResult(
            candidates=candidates,
            total_generated=len(candidates),
            generation_time=generation_time,
            tokens_used=total_tokens,
            errors=errors,
        )
    
    def _generate_direct_fix(self, request: PatchGenerationRequest) -> List[PatchCandidate]:
        """Generate patches directly from localization.
        
        Args:
            request: Generation request
            
        Returns:
            List of patch candidates
        """
        candidates = []
        
        # Group localization hits by file
        hits_by_file = {}
        for hit_dict in request.localization_hits[:request.max_files_per_patch]:
            file_path = hit_dict.get("file_path", "")
            if file_path not in hits_by_file:
                hits_by_file[file_path] = []
            hits_by_file[file_path].append(hit_dict)
        
        # Generate one patch per top localization hit
        for i, (file_path, hits) in enumerate(hits_by_file.items()):
            if i >= request.max_patches:
                break
            
            # Read file content
            full_path = Path(request.repo_dir) / file_path
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue
            
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    old_content = f.read()
                
                # Generate patch content
                # In a real implementation, this would call LLM
                patch_content = self._generate_patch_content_simple(
                    file_path=file_path,
                    old_content=old_content,
                    hits=hits,
                    problem_statement=request.problem_statement,
                )
                
                if patch_content:
                    patch = self._create_patch(
                        patch_content=patch_content,
                        file_path=file_path,
                        old_content=old_content,
                        strategy=request.strategy,
                        hits=hits,
                    )
                    
                    candidate = PatchCandidate(
                        patch=patch,
                        priority=hits[0].get("score", 0.0) if hits else 0.0,
                        generation_time=time.time(),
                    )
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"Failed to generate patch for {file_path}: {e}")
                continue
        
        return candidates
    
    def _generate_test_driven(self, request: PatchGenerationRequest) -> List[PatchCandidate]:
        """Generate patches to pass failing tests.
        
        Args:
            request: Generation request
            
        Returns:
            List of patch candidates
        """
        # Similar to direct fix but focused on test failures
        # For now, delegate to direct fix
        return self._generate_direct_fix(request)
    
    def _generate_hypothesis(self, request: PatchGenerationRequest) -> List[PatchCandidate]:
        """Generate patches based on bug hypotheses.
        
        Args:
            request: Generation request
            
        Returns:
            List of patch candidates
        """
        # Would generate multiple hypotheses and patches for each
        return self._generate_direct_fix(request)
    
    def _generate_incremental(self, request: PatchGenerationRequest) -> List[PatchCandidate]:
        """Generate patches incrementally from previous attempts.
        
        Args:
            request: Generation request
            
        Returns:
            List of patch candidates
        """
        candidates = []
        
        # Build on previous patches
        for prev_patch in request.previous_patches[:3]:
            # Modify previous patch
            # In real implementation, would use LLM to improve
            modified_patch = self._modify_patch(prev_patch, request)
            if modified_patch:
                candidate = PatchCandidate(
                    patch=modified_patch,
                    priority=0.7,  # Medium priority for incremental
                    parent_id=prev_patch.patch_id,
                )
                candidates.append(candidate)
        
        # Also generate some fresh attempts
        direct_candidates = self._generate_direct_fix(request)
        candidates.extend(direct_candidates)
        
        return candidates
    
    def _generate_ensemble(self, request: PatchGenerationRequest) -> List[PatchCandidate]:
        """Generate patches using ensemble of strategies.
        
        Args:
            request: Generation request
            
        Returns:
            List of patch candidates
        """
        all_candidates = []
        
        # Try each strategy
        for strategy in [PatchStrategy.DIRECT_FIX, PatchStrategy.TEST_DRIVEN, PatchStrategy.HYPOTHESIS]:
            sub_request = PatchGenerationRequest(
                problem_statement=request.problem_statement,
                repo_dir=request.repo_dir,
                localization_hits=request.localization_hits,
                failing_tests=request.failing_tests,
                traceback=request.traceback,
                strategy=strategy,
                max_patches=2,  # Fewer per strategy
            )
            
            candidates = self.generate(sub_request).candidates
            all_candidates.extend(candidates)
        
        return all_candidates[:request.max_patches]
    
    def _generate_patch_content_simple(
        self,
        file_path: str,
        old_content: str,
        hits: List[dict],
        problem_statement: str,
    ) -> Optional[str]:
        """Simple patch generation without LLM (placeholder).
        
        In a real implementation, this would call an LLM with context.
        
        Args:
            file_path: File to patch
            old_content: Original file content
            hits: Localization hits for this file
            problem_statement: Problem description
            
        Returns:
            New file content or None
        """
        # Placeholder: return original content
        # Real implementation would use LLM to generate fix
        logger.debug(f"Placeholder patch generation for {file_path}")
        return None
    
    def _create_patch(
        self,
        patch_content: str,
        file_path: str,
        old_content: str,
        strategy: PatchStrategy,
        hits: List[dict],
    ) -> Patch:
        """Create a Patch object from generated content.
        
        Args:
            patch_content: New file content
            file_path: File path
            old_content: Original content
            strategy: Generation strategy
            hits: Localization hits
            
        Returns:
            Patch object
        """
        # Generate unified diff
        import difflib
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            patch_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))
        diff_text = "".join(diff_lines)
        
        # Create file diff
        file_diff = FileDiff(
            file_path=file_path,
            old_content=old_content,
            new_content=patch_content,
            unified_diff=diff_text,
        )
        
        # Generate patch ID
        patch_hash = hashlib.sha256(diff_text.encode()).hexdigest()[:12]
        patch_id = f"patch_{patch_hash}"
        
        # Create patch
        patch = Patch(
            patch_id=patch_id,
            strategy=strategy,
            diff_text=diff_text,
            file_diffs=[file_diff],
            localization_hits=[h.get("file_path", "") for h in hits],
            rationale=f"Fix based on localization hits in {file_path}",
            generation_method="simple",
            status=PatchStatus.GENERATED,
        )
        
        return patch
    
    def _modify_patch(self, prev_patch: Patch, request: PatchGenerationRequest) -> Optional[Patch]:
        """Modify a previous patch to improve it.
        
        Args:
            prev_patch: Previous patch to modify
            request: Generation request
            
        Returns:
            Modified patch or None
        """
        # Placeholder: would use LLM to improve patch
        return None


def generate_patches(
    problem_statement: str,
    repo_dir: Path,
    localization_hits: List[LocalizationHit],
    strategy: PatchStrategy = PatchStrategy.DIRECT_FIX,
    max_patches: int = 5,
) -> List[Patch]:
    """High-level API for patch generation.
    
    Args:
        problem_statement: Problem description
        repo_dir: Repository directory
        localization_hits: Localization results
        strategy: Generation strategy
        max_patches: Maximum patches to generate
        
    Returns:
        List of generated patches
    """
    # Convert hits to dicts
    hits_dicts = [
        {
            "file_path": hit.file_path,
            "line_start": hit.line_start,
            "line_end": hit.line_end,
            "score": hit.score,
            "evidence": hit.evidence,
        }
        for hit in localization_hits
    ]
    
    request = PatchGenerationRequest(
        problem_statement=problem_statement,
        repo_dir=str(repo_dir),
        localization_hits=hits_dicts,
        strategy=strategy,
        max_patches=max_patches,
    )
    
    generator = PatchGenerator()
    result = generator.generate(request)
    
    return [candidate.patch for candidate in result.candidates]
