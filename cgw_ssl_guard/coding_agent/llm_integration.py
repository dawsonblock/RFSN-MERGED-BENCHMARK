"""LLM integration for the CGW coding agent.

This module provides LLM-powered proposal generators that integrate with
the existing RFSN controller's multi-model ensemble. The LLM is used to:

1. Generate patches based on test failures
2. Analyze tracebacks for root cause
3. Suggest next actions based on context

IMPORTANT: The LLM PROPOSES candidates, it does not DECIDE. The final
decision is made by the thalamic gate based on scored competition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..types import Candidate
from .action_types import CodingAction
from .proposal_generators import ProposalContext, ProposalGenerator

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 60


class LLMPatchGenerator(ProposalGenerator):
    """LLM-powered patch generation.
    
    Uses the model to generate patches based on:
    - Test failures and tracebacks
    - File contents
    - Previous patch attempts
    
    The generator produces GENERATE_PATCH action candidates with the
    generated diff in the parameters.
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm_caller: Optional[Callable[..., str]] = None,
    ):
        super().__init__("llm_patch")
        self.config = config or LLMConfig()
        self.llm_caller = llm_caller
        self._patch_cache: Dict[str, str] = {}
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        """Generate patch proposals using LLM."""
        # Only generate patches after analysis
        if context.last_action != CodingAction.ANALYZE_FAILURE:
            return []
        
        # Check if we have failing tests to fix
        if not context.failing_tests and not context.test_output:
            return []
        
        # Generate patch via LLM
        try:
            patch = self._generate_patch(context)
            if patch:
                # IMPORTANT: emit APPLY_PATCH directly.
                # The executor can apply the diff in a single, auditable action.
                # This avoids a no-op GENERATE_PATCH cycle and keeps the CGW loop purely
                # "propose -> select -> execute".
                return [self._make_candidate(
                    CodingAction.APPLY_PATCH,
                    saliency=0.85,
                    urgency=0.6,
                    parameters={"diff": patch, "source": "llm"},
                    context={"failing_tests": context.failing_tests[:5]},
                )]
        except Exception as e:
            logger.warning(f"LLM patch generation failed: {e}")
        
        return []
    
    def _generate_patch(self, context: ProposalContext) -> Optional[str]:
        """Call LLM to generate a patch."""
        if self.llm_caller is None:
            # Return mock patch for testing
            logger.debug("No LLM caller configured, returning mock patch")
            return self._generate_mock_patch(context)
        
        # Build prompt
        prompt = self._build_patch_prompt(context)
        
        # Call LLM
        response = self.llm_caller(
            prompt=prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        # Extract diff from response
        return self._extract_diff(response)
    
    def _build_patch_prompt(self, context: ProposalContext) -> str:
        """Build the LLM prompt for patch generation."""
        return f"""You are a code repair agent. Generate a minimal patch to fix the failing tests.

## Failing Tests
{json.dumps(context.failing_tests, indent=2)}

## Test Output
```
{context.test_output[:4000]}
```

## Goal
{context.goal}

## Instructions
1. Analyze the test failures
2. Identify the root cause
3. Generate a minimal unified diff to fix the issue

Return ONLY the unified diff, nothing else. Format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,3 +1,3 @@
 unchanged
-old line
+new line
 unchanged
```
"""
    
    def _extract_diff(self, response: str) -> Optional[str]:
        """Extract unified diff from LLM response."""
        # Look for diff block
        if "```diff" in response:
            start = response.find("```diff") + 7
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Look for --- a/ pattern
        if "--- a/" in response:
            lines = response.split("\n")
            diff_lines = []
            in_diff = False
            for line in lines:
                if line.startswith("--- a/"):
                    in_diff = True
                if in_diff:
                    if line.startswith("```") or (not line and not diff_lines):
                        continue
                    diff_lines.append(line)
                    if line.startswith("+++ ") and not line.startswith("+++"):
                        break
            if diff_lines:
                return "\n".join(diff_lines)
        
        return None
    
    def _generate_mock_patch(self, context: ProposalContext) -> str:
        """Generate a mock patch for testing."""
        # This is used when no LLM is configured
        return """--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 def foo():
-    return None
+    return 42
"""


class LLMAnalysisGenerator(ProposalGenerator):
    """LLM-powered failure analysis.
    
    Uses the model to analyze test failures and suggest the most
    relevant next action with appropriate urgency/saliency.
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm_caller: Optional[Callable[..., str]] = None,
    ):
        super().__init__("llm_analysis")
        self.config = config or LLMConfig()
        self.llm_caller = llm_caller
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        """Generate analysis-based proposals using LLM."""
        # Only analyze after failed tests
        if context.last_action != CodingAction.RUN_TESTS:
            return []
        if context.tests_passing:
            return []
        
        candidates = []
        
        # Always propose standard analysis
        candidates.append(self._make_candidate(
            CodingAction.ANALYZE_FAILURE,
            saliency=0.8,
            urgency=0.5,
            context={"source": "llm_analysis"},
        ))
        
        # If we detect specific patterns, add targeted proposals
        if context.test_output:
            patterns = self._detect_patterns(context.test_output)
            
            if "import_error" in patterns:
                candidates.append(self._make_candidate(
                    CodingAction.INSPECT_FILES,
                    saliency=0.75,
                    urgency=0.6,
                    surprise=0.7,  # Import errors can be surprising
                    context={"pattern": "import_error", "source": "llm_analysis"},
                ))
            
            if "syntax_error" in patterns:
                candidates.append(self._make_candidate(
                    CodingAction.ANALYZE_TRACEBACK,
                    saliency=0.9,
                    urgency=0.9,  # Syntax errors are urgent
                    context={"pattern": "syntax_error", "source": "llm_analysis"},
                ))
        
        return candidates
    
    def _detect_patterns(self, output: str) -> List[str]:
        """Detect error patterns in test output."""
        patterns = []
        output_lower = output.lower()
        
        if "importerror" in output_lower or "modulenotfounderror" in output_lower:
            patterns.append("import_error")
        if "syntaxerror" in output_lower:
            patterns.append("syntax_error")
        if "typeerror" in output_lower:
            patterns.append("type_error")
        if "assertionerror" in output_lower:
            patterns.append("assertion_error")
        if "timeout" in output_lower:
            patterns.append("timeout")
        if "memory" in output_lower and "error" in output_lower:
            patterns.append("memory_error")
        
        return patterns


class LLMDecisionAdvisor(ProposalGenerator):
    """LLM-powered decision advisor.
    
    Uses the model to suggest the best next action based on the full
    context of the repair session. This is a meta-generator that can
    propose any action type based on LLM reasoning.
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        llm_caller: Optional[Callable[..., str]] = None,
    ):
        super().__init__("llm_advisor")
        self.config = config or LLMConfig()
        self.llm_caller = llm_caller
    
    def generate(self, context: ProposalContext) -> List[Candidate]:
        """Generate action proposals based on LLM reasoning."""
        if self.llm_caller is None:
            return []
        
        try:
            recommendation = self._get_recommendation(context)
            if recommendation:
                action = self._map_recommendation_to_action(recommendation)
                return [self._make_candidate(
                    action,
                    saliency=0.7,  # Lower than domain-specific generators
                    urgency=0.4,
                    context={"recommendation": recommendation, "source": "llm_advisor"},
                )]
        except Exception as e:
            logger.debug(f"LLM advisor failed: {e}")
        
        return []
    
    def _get_recommendation(self, context: ProposalContext) -> Optional[str]:
        """Query LLM for action recommendation."""
        prompt = f"""You are a coding agent decision advisor.

Current state:
- Last action: {context.last_action.value if context.last_action else 'None'}
- Tests passing: {context.tests_passing}
- Patches applied: {context.patches_applied}
- Failing tests: {len(context.failing_tests)}

What should be the next action? Choose from:
- RUN_TESTS: Run the test suite
- ANALYZE_FAILURE: Analyze test failures
- GENERATE_PATCH: Generate a code patch
- APPLY_PATCH: Apply a pending patch
- VALIDATE: Run validation checks
- FINALIZE: Finish successfully
- ABORT: Stop due to issues

Respond with just the action name."""
        
        response = self.llm_caller(
            prompt=prompt,
            model=self.config.model,
            temperature=0.1,  # Low temp for decision
            max_tokens=50,
        )
        
        return response.strip().upper()
    
    def _map_recommendation_to_action(self, recommendation: str) -> CodingAction:
        """Map LLM recommendation to CodingAction."""
        for action in CodingAction:
            if action.value in recommendation:
                return action
        return CodingAction.IDLE
