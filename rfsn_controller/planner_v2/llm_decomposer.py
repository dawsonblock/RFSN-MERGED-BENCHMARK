"""LLM Decomposer - Intelligent goal decomposition using LLM.

Uses LLM to decompose high-level goals into structured, atomic steps.
Falls back to pattern-based decomposition on LLM failure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schema import Plan, RiskLevel, Step
from .tool_registry import get_tool_registry


@dataclass
class DecompositionConfig:
    """Configuration for LLM decomposition."""
    
    temperature: float = 0.3
    max_tokens: int = 2000
    max_retries: int = 2
    timeout_sec: float = 30.0
    min_steps: int = 2
    max_steps: int = 10
    require_verification: bool = True


from datetime import UTC

from .model_selector import ModelSelector

# Type alias for LLM call function: (prompt, temperature, max_tokens, model_id) -> response
LLMCallFn = Callable[[str, float, int, str | None], str]


DECOMPOSITION_SYSTEM_PROMPT = """You are a software engineering planner. Your job is to decompose high-level goals into atomic, executable steps.

RULES:
1. Each step must be independently executable
2. Steps must have clear success criteria
3. Higher-risk steps (modifying core files) need rollback hints
4. Each step must specify which files it may touch
5. Steps should be ordered by dependencies
6. Include verification commands where applicable

FORBIDDEN FILES (never include):
- controller.py, safety.py, rules.py
- *.env, secrets/*, credentials/*
- node_modules/*, __pycache__/*

RISK LEVELS:
- LOW: Read-only analysis, documentation, simple edits
- MED: Modifying test files, adding new files
- HIGH: Modifying core logic, refactoring existing code

OUTPUT FORMAT (JSON array):
[
  {
    "step_id": "analyze-failure",
    "title": "Analyze test failure",
    "intent": "Understand why the test is failing by examining the error message and test code",
    "allowed_files": ["tests/test_example.py", "src/module.py"],
    "success_criteria": "Root cause of failure identified",
    "dependencies": [],
    "verify": "python -c 'import src.module'",
    "risk_level": "LOW",
    "rollback_hint": "",
    "hypothesis": "The import fails because the module is not in PYTHONPATH"
  },
  ...
]
"""



DECOMPOSITION_USER_PROMPT = """Goal: {goal}

Context:
- Repository type: {repo_type}
- Primary language: {language}
- Test command: {test_cmd}
- Failing tests: {failing_tests}
- Failure type: {failure_type}
- Additional context: {extra_context}

Decompose this goal into {min_steps}-{max_steps} atomic, ordered steps.
Output ONLY valid JSON array of steps, no explanation."""

# Failure-specific prompts (Upgrade 5)
BUILD_FIX_PROMPT = """Goal: Fix Build Error - {goal}

Context:
- Build command failed.
- Language: {language}
- Failure type: {failure_type}
- Error: {extra_context}

Focus on:
1. Identifying missing dependencies or syntax errors.
2. Checking build configuration (setup.py, package.json, etc).
3. Verifying the build environment.

Hypothesize the root cause in the 'hypothesis' field.
"""

TEST_FIX_PROMPT = """Goal: Fix Test Failure - {goal}

Context:
- Tests failed: {failing_tests}
- Language: {language}
- Failure type: {failure_type}
- Error/Trace: {extra_context}

Focus on:
1. Analyzing the failure trace.
2. Reproducing the failure with a focused test run.
3. Modifying the code to fix the logic.
4. Verifying the fix.

Hypothesize why the test failed in the 'hypothesis' field.
"""


class LLMDecomposer:
    """Uses LLM to decompose goals into structured plans."""
    
    def __init__(
        self,
        llm_call: LLMCallFn | None = None,
        config: DecompositionConfig | None = None,
        model_selector: ModelSelector | None = None,
        firewall_warnings: list[str] | None = None,
    ):
        """Initialize the decomposer.
        
        Args:
            llm_call: Function to call LLM. Signature: (prompt, temp, tokens, model_id) -> response
            config: Decomposition configuration.
            model_selector: Learned model arbitration.
            firewall_warnings: Warnings to inject into prompt.
        """
        self._llm_call = llm_call
        self._config = config or DecompositionConfig()
        self._model_selector = model_selector
        self._firewall_warnings = firewall_warnings or []
    
    def decompose(
        self,
        goal: str,
        context: dict[str, Any],
        plan_id: str,
    ) -> Plan | None:
        """Decompose a goal into a structured plan using LLM.
        
        Args:
            goal: The high-level goal description.
            context: Execution context (repo_type, language, test_cmd, etc.)
            plan_id: ID for the generated plan.
            
        Returns:
            Plan if successful, None if LLM fails or returns invalid response.
        """
        if self._llm_call is None:
            return None
        
        # Select model if selector available
        model_id = None
        if self._model_selector:
            # Detect context from arguments
            context.get("repo_type", "unknown")
            language = context.get("language", "unknown")
            failure_type = context.get("failure_type", "unknown")
            
            # Select top model
            options = self._model_selector.select_model(
                goal_type="decompose", # Generic goal for now
                failure_type=failure_type,
                language=language,
                k=1,
            )
            if options:
                model_id = options[0].id

        prompt = self._build_decomposition_prompt(goal, context)
        
        for attempt in range(self._config.max_retries):
            start_time = 0 # In real implementation use time.monotonic()
            try:
                # Assuming time module imported or mocked
                import time
                start_time = time.monotonic()
                
                response = self._llm_call(
                    prompt,
                    self._config.temperature,
                    self._config.max_tokens,
                    model_id,
                )
                
                latency_ms = int((time.monotonic() - start_time) * 1000)
                
                steps = self._parse_llm_response(response)
                
                if steps and self._validate_decomposition(steps):
                    # Record success
                    if self._model_selector and model_id:
                        self._model_selector.record_outcome(
                            model_id, "decompose", failure_type, language, True, latency_ms
                        )
                    return self._build_plan(plan_id, goal, steps, context)
                
                # Record logical failure (invalid response)
                if self._model_selector and model_id:
                    self._model_selector.record_outcome(
                        model_id, "decompose", failure_type, language, False, latency_ms
                    )
                    
            except Exception:
                # LLM call failed or crashed
                if self._model_selector and model_id:
                     # Estimate latency?
                     self._model_selector.record_outcome(
                        model_id, "decompose", failure_type, language, False, 0
                    )
                continue
        
        return None
    
    def _build_decomposition_prompt(self, goal: str, context: dict[str, Any]) -> str:
        """Build the decomposition prompt for the LLM.
        
        Args:
            goal: The goal to decompose.
            context: Execution context.
            
        Returns:
            Formatted prompt string.
        """
        # Get allowed tools
        registry = get_tool_registry()
        tools = registry.list_tools()
        tool_desc = "\\n".join([f"- {t.name}: {t.command_template} ({t.description})" for t in tools])
        
        # Extract context values
        repo_type = context.get("repo_type", "unknown")
        language = context.get("language", "python")
        test_cmd = context.get("test_cmd", "pytest")
        failing_tests = context.get("failing_tests", [])
        
        # Build extra context string
        extra_parts = []
        if context.get("target_file"):
            extra_parts.append(f"Target file: {context['target_file']}")
        if context.get("error_message"):
            extra_parts.append(f"Error: {context['error_message'][:200]}")
        if context.get("issue_text"):
            extra_parts.append(f"Issue: {context['issue_text'][:300]}")
            
        # Add regression warnings if firewall available
        if self._model_selector and hasattr(self, "_firewall_warnings") and self._firewall_warnings:
             extra_parts.append(f"WARNINGS: {'; '.join(self._firewall_warnings)}")
            
        extra_context = "; ".join(extra_parts) if extra_parts else "None"
        
        failure_type = context.get("failure_type", "unknown")
        
        # Select template based on failure type (Upgrade 5)
        if failure_type == "build_error" or "Build Error" in goal:
            template = BUILD_FIX_PROMPT
        elif failure_type == "test_failure" or context.get("failing_tests"):
            template = TEST_FIX_PROMPT
        else:
            template = DECOMPOSITION_USER_PROMPT
            
        user_prompt = template.format(
            goal=goal,
            repo_type=repo_type,
            language=language,
            test_cmd=test_cmd,
            failing_tests=", ".join(failing_tests) if failing_tests else "None",
            failure_type=failure_type,
            extra_context=extra_context,
            min_steps=self._config.min_steps,
            max_steps=self._config.max_steps,
        )
        
        system_prompt = DECOMPOSITION_SYSTEM_PROMPT.replace(
            "6. Include verification commands where applicable",
            f"6. Include verification commands where applicable. VALID TOOLS:\\n{tool_desc}"
        )
        
        return f"{system_prompt}\\n\\n{user_prompt}"
    
    def _parse_llm_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into step dictionaries.
        
        Args:
            response: Raw LLM response.
            
        Returns:
            List of step dictionaries.
        """
        # Try to extract JSON from response
        # Handle cases where LLM adds explanation before/after JSON
        
        # Look for JSON array
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire response as JSON
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _validate_decomposition(self, steps: list[dict[str, Any]]) -> bool:
        """Validate the decomposition meets requirements.
        
        Args:
            steps: List of step dictionaries from LLM.
            
        Returns:
            True if valid, False otherwise.
        """
        if len(steps) < self._config.min_steps:
            return False
        if len(steps) > self._config.max_steps:
            return False
        
        required_fields = ["step_id", "title", "intent", "allowed_files", "success_criteria"]
        
        for step in steps:
            # Check required fields
            for field in required_fields:
                if field not in step:
                    return False
            
            # Check for forbidden files
            for f in step.get("allowed_files", []):
                if any(forbidden in f for forbidden in ["controller.py", "safety.py", ".env", "secrets/"]):
                    return False
            
            # Check verification command validity
            if self._config.require_verification and step.get("verify"):
                if not get_tool_registry().validate_command(step["verify"]):
                    return False
        
        return True
    
    def _build_plan(
        self,
        plan_id: str,
        goal: str,
        step_dicts: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> Plan:
        """Build a Plan from validated step dictionaries.
        
        Args:
            plan_id: Plan identifier.
            goal: Original goal.
            step_dicts: Validated step dictionaries.
            context: Execution context.
            
        Returns:
            Constructed Plan.
        """
        from datetime import datetime
        
        steps = []
        for step_dict in step_dicts:
            risk_str = step_dict.get("risk_level", "LOW").upper()
            try:
                risk = RiskLevel(risk_str)
            except ValueError:
                risk = RiskLevel.LOW
            
            step = Step(
                step_id=step_dict["step_id"],
                title=step_dict["title"],
                intent=step_dict["intent"],
                allowed_files=step_dict.get("allowed_files", []),
                success_criteria=step_dict.get("success_criteria", ""),
                dependencies=step_dict.get("dependencies", []),
                inputs=step_dict.get("inputs", []),
                verify=step_dict.get("verify", ""),
                risk_level=risk,
                rollback_hint=step_dict.get("rollback_hint", ""),
                hypothesis=step_dict.get("hypothesis", ""),
            )
            steps.append(step)
        
        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            created_at=datetime.now(UTC).isoformat(),
            assumptions=["LLM-generated decomposition"],
            constraints=context.get("constraints", []),
        )
    
    def get_prompt_template(self) -> str:
        """Get the system prompt template for debugging/inspection."""
        return DECOMPOSITION_SYSTEM_PROMPT


class DecompositionFallback:
    """Manages fallback between LLM and pattern-based decomposition."""
    
    def __init__(
        self,
        llm_decomposer: LLMDecomposer | None = None,
        pattern_decomposer: Callable | None = None,
    ):
        """Initialize with decomposers.
        
        Args:
            llm_decomposer: LLM-based decomposer.
            pattern_decomposer: Pattern-based fallback function.
        """
        self._llm = llm_decomposer
        self._pattern = pattern_decomposer
        self._last_source = "none"
    
    def decompose(
        self,
        goal: str,
        context: dict[str, Any],
        plan_id: str,
    ) -> Plan | None:
        """Decompose with fallback.
        
        Tries LLM first, falls back to pattern-based.
        
        Args:
            goal: The goal to decompose.
            context: Execution context.
            plan_id: Plan identifier.
            
        Returns:
            Plan if either method succeeds, None otherwise.
        """
        # Try LLM first
        if self._llm:
            plan = self._llm.decompose(goal, context, plan_id)
            if plan:
                self._last_source = "llm"
                return plan
        
        # Fall back to pattern-based
        if self._pattern:
            plan = self._pattern(goal, context, plan_id)
            if plan:
                self._last_source = "pattern"
                return plan
        
        self._last_source = "none"
        return None
    
    @property
    def last_source(self) -> str:
        """Get the source of the last successful decomposition."""
        return self._last_source
