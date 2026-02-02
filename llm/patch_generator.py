"""
LLM-Powered Patch Generator

Integrates LLM client with patch generation strategies.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from patch.types import Patch, PatchCandidate, PatchStrategy, PatchStatus, FileDiff
from localize.types import LocalizationHit
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)

# Import LLM components
try:
    from llm.client import get_llm_client, LLMClient, LLMResponse
    from llm.prompts import PatchPromptTemplates, build_context_from_localization, PromptContext
    LLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM client not available: {e}")
    LLM_AVAILABLE = False


class LLMPatchGenerator:
    """Generate patches using LLM"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize with optional LLM client"""
        self.llm_client = llm_client or (get_llm_client() if LLM_AVAILABLE else None)
        self.total_tokens = 0
        self.total_cost = 0.0
    
    async def generate_direct_fix(
        self,
        problem: str,
        repo_path: str,
        localization_hits: List[Dict[str, Any]],
        max_patches: int = 3
    ) -> List[Patch]:
        """Generate patches using direct fix strategy"""
        if not self.llm_client:
            logger.warning("No LLM client available, returning empty list")
            return []
        
        patches = []
        
        for i, hit in enumerate(localization_hits[:max_patches]):
            try:
                file_path = hit.get("file_path", "")
                line_start = hit.get("line_start", 0)
                line_end = hit.get("line_end", 0)
                evidence = hit.get("evidence", "")
                
                # Build context
                ctx = build_context_from_localization(
                    problem=problem,
                    repo_path=repo_path,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    evidence=evidence
                )
                
                # Generate prompt
                system, user = PatchPromptTemplates.direct_fix_prompt(ctx)
                
                # Get LLM response
                response = await self.llm_client.complete_with_retry(
                    prompt=user,
                    system=system,
                    max_tokens=2048
                )
                
                # Track usage
                self.total_tokens += response.tokens_used
                self.total_cost += response.cost_usd
                
                # Extract patch from response
                patch = self._extract_patch_from_response(
                    response.content,
                    file_path,
                    strategy=PatchStrategy.DIRECT_FIX
                )
                
                if patch:
                    patches.append(patch)
                    logger.info(f"Generated patch {i+1}/{max_patches} for {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate patch for hit {i}: {e}")
        
        logger.info(f"Generated {len(patches)} patches (tokens: {self.total_tokens}, cost: ${self.total_cost:.4f})")
        return patches
    
    async def generate_test_driven(
        self,
        problem: str,
        repo_path: str,
        localization_hits: List[Dict[str, Any]],
        test_output: Optional[str] = None,
        error_trace: Optional[str] = None,
        max_patches: int = 2
    ) -> List[Patch]:
        """Generate patches using test-driven strategy"""
        if not self.llm_client:
            return []
        
        patches = []
        
        for i, hit in enumerate(localization_hits[:max_patches]):
            try:
                file_path = hit.get("file_path", "")
                line_start = hit.get("line_start", 0)
                line_end = hit.get("line_end", 0)
                evidence = hit.get("evidence", "")
                
                # Build context with test info
                ctx = build_context_from_localization(
                    problem=problem,
                    repo_path=repo_path,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    evidence=evidence,
                    error_trace=error_trace
                )
                ctx.test_output = test_output
                
                # Generate prompt
                system, user = PatchPromptTemplates.test_driven_prompt(ctx)
                
                # Get LLM response
                response = await self.llm_client.complete_with_retry(
                    prompt=user,
                    system=system,
                    max_tokens=2048
                )
                
                self.total_tokens += response.tokens_used
                self.total_cost += response.cost_usd
                
                # Extract patch
                patch = self._extract_patch_from_response(
                    response.content,
                    file_path,
                    strategy=PatchStrategy.TEST_DRIVEN
                )
                
                if patch:
                    patches.append(patch)
                
            except Exception as e:
                logger.error(f"Failed to generate test-driven patch {i}: {e}")
        
        return patches
    
    async def generate_hypothesis_driven(
        self,
        problem: str,
        repo_path: str,
        localization_hits: List[Dict[str, Any]],
        error_trace: Optional[str] = None,
        max_patches: int = 2
    ) -> List[Patch]:
        """Generate patches using hypothesis-driven strategy"""
        if not self.llm_client:
            return []
        
        patches = []
        
        for i, hit in enumerate(localization_hits[:max_patches]):
            try:
                file_path = hit.get("file_path", "")
                line_start = hit.get("line_start", 0)
                line_end = hit.get("line_end", 0)
                evidence = hit.get("evidence", "")
                
                ctx = build_context_from_localization(
                    problem=problem,
                    repo_path=repo_path,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    evidence=evidence,
                    error_trace=error_trace
                )
                
                system, user = PatchPromptTemplates.hypothesis_driven_prompt(ctx)
                
                response = await self.llm_client.complete_with_retry(
                    prompt=user,
                    system=system,
                    max_tokens=3072  # More tokens for hypothesis analysis
                )
                
                self.total_tokens += response.tokens_used
                self.total_cost += response.cost_usd
                
                patch = self._extract_patch_from_response(
                    response.content,
                    file_path,
                    strategy=PatchStrategy.HYPOTHESIS
                )
                
                if patch:
                    patches.append(patch)
                
            except Exception as e:
                logger.error(f"Failed to generate hypothesis patch {i}: {e}")
        
        return patches
    
    def _extract_patch_from_response(
        self,
        content: str,
        file_path: str,
        strategy: PatchStrategy
    ) -> Optional[Patch]:
        """Extract unified diff patch from LLM response"""
        
        # Look for diff blocks in response
        diff_pattern = r'```diff\n(.*?)\n```'
        matches = re.findall(diff_pattern, content, re.DOTALL)
        
        if not matches:
            # Try without code fence
            if content.strip().startswith('---') or content.strip().startswith('@@'):
                diff_text = content.strip()
            else:
                logger.warning("No diff found in LLM response")
                return None
        else:
            diff_text = matches[0].strip()
        
        if not diff_text:
            return None
        
        # Create FileDiff
        file_diff = FileDiff(
            file_path=file_path,
            diff_text=diff_text,
            additions=len([l for l in diff_text.split('\n') if l.startswith('+')]),
            deletions=len([l for l in diff_text.split('\n') if l.startswith('-')]),
            hunks=self._parse_hunks(diff_text)
        )
        
        # Create Patch
        patch = Patch(
            patch_id=f"patch_{hash(diff_text) & 0xffffffff:08x}",
            strategy=strategy,
            files=[file_diff],
            description=f"LLM-generated patch using {strategy.value} strategy",
            status=PatchStatus.GENERATED,
            metadata={
                "llm_generated": True,
                "source": "LLMPatchGenerator"
            }
        )
        
        return patch
    
    def _parse_hunks(self, diff_text: str) -> List[Dict[str, Any]]:
        """Parse diff hunks from unified diff"""
        hunks = []
        current_hunk = None
        
        for line in diff_text.split('\n'):
            if line.startswith('@@'):
                # New hunk
                if current_hunk:
                    hunks.append(current_hunk)
                
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                parts = line.split('@@')[1].strip().split()
                old_range = parts[0][1:].split(',')  # Remove '-'
                new_range = parts[1][1:].split(',')  # Remove '+'
                
                current_hunk = {
                    "old_start": int(old_range[0]),
                    "old_count": int(old_range[1]) if len(old_range) > 1 else 1,
                    "new_start": int(new_range[0]),
                    "new_count": int(new_range[1]) if len(new_range) > 1 else 1,
                    "lines": []
                }
            elif current_hunk:
                current_hunk["lines"].append(line)
        
        if current_hunk:
            hunks.append(current_hunk)
        
        return hunks
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "client_stats": self.llm_client.stats.__dict__ if self.llm_client else {}
        }


async def generate_patches_with_llm(
    problem: str,
    repo_path: str,
    localization_hits: List[Dict[str, Any]],
    strategy: PatchStrategy = PatchStrategy.DIRECT_FIX,
    test_output: Optional[str] = None,
    error_trace: Optional[str] = None,
    max_patches: int = 3
) -> List[Patch]:
    """
    Generate patches using LLM
    
    High-level function for patch generation with LLM.
    """
    generator = LLMPatchGenerator()
    
    if strategy == PatchStrategy.DIRECT_FIX:
        patches = await generator.generate_direct_fix(
            problem, repo_path, localization_hits, max_patches
        )
    elif strategy == PatchStrategy.TEST_DRIVEN:
        patches = await generator.generate_test_driven(
            problem, repo_path, localization_hits, test_output, error_trace, max_patches
        )
    elif strategy == PatchStrategy.HYPOTHESIS:
        patches = await generator.generate_hypothesis_driven(
            problem, repo_path, localization_hits, error_trace, max_patches
        )
    else:
        logger.warning(f"Strategy {strategy} not yet implemented with LLM")
        patches = []
    
    # Log usage stats
    stats = generator.get_usage_stats()
    logger.info(f"LLM usage: {stats}")
    
    return patches


if __name__ == "__main__":
    # Test
    async def test():
        hits = [
            {
                "file_path": "test.py",
                "line_start": 5,
                "line_end": 10,
                "evidence": "Function returns wrong value"
            }
        ]
        
        patches = await generate_patches_with_llm(
            problem="Fix the add function",
            repo_path="/tmp",
            localization_hits=hits,
            strategy=PatchStrategy.DIRECT_FIX,
            max_patches=1
        )
        
        print(f"Generated {len(patches)} patches")
        for patch in patches:
            print(f"Patch {patch.patch_id}:")
            for file_diff in patch.files:
                print(f"  {file_diff.file_path}: +{file_diff.additions}/-{file_diff.deletions}")
    
    asyncio.run(test())
