"""
Prompt templates for LLM-based patch generation

Provides structured prompts for different patch generation strategies:
- Direct fix from localization
- Test-driven development
- Hypothesis-driven debugging
- Incremental patching
- Ensemble patching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class PromptContext:
    """Context for patch generation prompts"""
    problem_statement: str
    repo_path: str
    file_path: str
    file_content: str
    line_start: int
    line_end: int
    localization_evidence: str
    error_trace: Optional[str] = None
    test_output: Optional[str] = None
    related_files: List[str] = None
    commit_history: Optional[str] = None


class PatchPromptTemplates:
    """Templates for patch generation prompts"""
    
    @staticmethod
    def direct_fix_prompt(ctx: PromptContext) -> tuple[str, str]:
        """Generate prompt for direct fix strategy"""
        
        system = """You are an expert software engineer specializing in bug fixing.
Your task is to analyze code and generate precise, minimal patches that fix bugs.

Guidelines:
- Provide ONLY the changed lines in unified diff format
- Make minimal, surgical changes
- Preserve code style and formatting
- Add comments only if necessary for clarity
- Ensure the fix doesn't break existing functionality
"""
        
        user = f"""# Bug Report

{ctx.problem_statement}

# Localized Bug Location

File: {ctx.file_path}
Lines: {ctx.line_start}-{ctx.line_end}
Evidence: {ctx.localization_evidence}

# Current Code

```python
{ctx.file_content}
```

{f'''# Error Trace

```
{ctx.error_trace}
```
''' if ctx.error_trace else ''}

{f'''# Test Output

```
{ctx.test_output}
```
''' if ctx.test_output else ''}

# Task

Generate a minimal patch in unified diff format that fixes this bug.
Focus on lines {ctx.line_start}-{ctx.line_end}.

Output format:
```diff
--- a/{ctx.file_path}
+++ b/{ctx.file_path}
@@ -X,Y +X,Y @@
 context line
-old line
+new line
 context line
```
"""
        return system, user
    
    @staticmethod
    def test_driven_prompt(ctx: PromptContext) -> tuple[str, str]:
        """Generate prompt for test-driven strategy"""
        
        system = """You are an expert in test-driven development and debugging.
Your task is to:
1. Understand the failing test
2. Identify the root cause
3. Generate a fix that makes the test pass
4. Ensure no other tests break
"""
        
        user = f"""# Failing Test

{ctx.problem_statement}

# Test Output

```
{ctx.test_output or 'No test output available'}
```

# Error Trace

```
{ctx.error_trace or 'No error trace available'}
```

# Code Under Test

File: {ctx.file_path}
Lines: {ctx.line_start}-{ctx.line_end}

```python
{ctx.file_content}
```

# Task

1. Analyze why the test is failing
2. Identify the bug in the code
3. Generate a minimal patch that fixes the failing test
4. Ensure your fix doesn't break other functionality

Provide your patch in unified diff format.
"""
        return system, user
    
    @staticmethod
    def hypothesis_driven_prompt(ctx: PromptContext) -> tuple[str, str]:
        """Generate prompt for hypothesis-driven strategy"""
        
        system = """You are an expert debugger using hypothesis-driven development.
Your task is to:
1. Form hypotheses about the bug's root cause
2. Analyze evidence for each hypothesis
3. Select the most likely cause
4. Generate a targeted fix
"""
        
        user = f"""# Bug Report

{ctx.problem_statement}

# Available Evidence

## Localization
File: {ctx.file_path}
Lines: {ctx.line_start}-{ctx.line_end}
Evidence: {ctx.localization_evidence}

## Code
```python
{ctx.file_content}
```

{f'''## Error Trace
```
{ctx.error_trace}
```
''' if ctx.error_trace else ''}

{f'''## Test Output
```
{ctx.test_output}
```
''' if ctx.test_output else ''}

# Task

1. List 2-3 hypotheses about what might be causing this bug
2. Analyze evidence for each hypothesis
3. Select the most likely root cause
4. Generate a patch that addresses the root cause

Format:
```
Hypotheses:
1. [Hypothesis 1]
2. [Hypothesis 2]
3. [Hypothesis 3]

Analysis:
[Your analysis]

Selected Hypothesis: [X]

Patch:
```diff
[Your patch in unified diff format]
```
```
"""
        return system, user
    
    @staticmethod
    def incremental_prompt(
        ctx: PromptContext,
        previous_attempts: List[str]
    ) -> tuple[str, str]:
        """Generate prompt for incremental patching"""
        
        system = """You are an expert at iterative debugging.
Previous fix attempts have failed. Your task is to:
1. Learn from previous failures
2. Identify what went wrong
3. Generate an improved fix
"""
        
        prev_attempts_str = "\n\n".join([
            f"## Attempt {i+1}\n```diff\n{attempt}\n```"
            for i, attempt in enumerate(previous_attempts)
        ])
        
        user = f"""# Bug Report

{ctx.problem_statement}

# Code Location

File: {ctx.file_path}
Lines: {ctx.line_start}-{ctx.line_end}

```python
{ctx.file_content}
```

# Previous Failed Attempts

{prev_attempts_str}

{f'''# Latest Test Output
```
{ctx.test_output}
```
''' if ctx.test_output else ''}

# Task

Analyze why previous attempts failed and generate an improved patch.
Learn from the mistakes and try a different approach.

Provide your improved patch in unified diff format.
"""
        return system, user
    
    @staticmethod
    def ensemble_prompt(
        ctx: PromptContext,
        candidate_patches: List[str]
    ) -> tuple[str, str]:
        """Generate prompt for ensemble strategy"""
        
        system = """You are an expert code reviewer evaluating multiple patch candidates.
Your task is to:
1. Review each candidate patch
2. Identify strengths and weaknesses
3. Synthesize the best elements into a single optimal patch
"""
        
        candidates_str = "\n\n".join([
            f"## Candidate {i+1}\n```diff\n{patch}\n```"
            for i, patch in enumerate(candidate_patches)
        ])
        
        user = f"""# Bug Report

{ctx.problem_statement}

# Code Location

File: {ctx.file_path}

```python
{ctx.file_content}
```

# Candidate Patches

{candidates_str}

# Task

1. Review each candidate patch
2. Identify the best approach
3. Synthesize an optimal patch combining the best elements

Provide your final patch in unified diff format.
"""
        return system, user
    
    @staticmethod
    def code_review_prompt(patch: str, original_code: str) -> tuple[str, str]:
        """Generate prompt for patch review"""
        
        system = """You are an expert code reviewer.
Your task is to review a patch and identify potential issues:
- Logic errors
- Edge cases not handled
- Breaking changes
- Code style violations
- Performance issues
"""
        
        user = f"""# Original Code

```python
{original_code}
```

# Proposed Patch

```diff
{patch}
```

# Task

Review this patch and provide:
1. Correctness score (0-100)
2. Potential issues (if any)
3. Recommendations for improvement

Format:
```json
{{
  "correctness_score": 85,
  "issues": [
    "Doesn't handle edge case X",
    "May break functionality Y"
  ],
  "recommendations": [
    "Add null check",
    "Consider error handling"
  ],
  "approved": false
}}
```
"""
        return system, user


def build_context_from_localization(
    problem: str,
    repo_path: str,
    file_path: str,
    line_start: int,
    line_end: int,
    evidence: str,
    error_trace: Optional[str] = None,
    max_context_lines: int = 50
) -> PromptContext:
    """Build prompt context from localization hit"""
    
    # Read file content with context
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Get context around bug location
        start_idx = max(0, line_start - 10)
        end_idx = min(len(lines), line_end + 10)
        
        # Limit total context size
        if end_idx - start_idx > max_context_lines:
            # Prioritize the bug location
            bug_size = line_end - line_start
            context_budget = (max_context_lines - bug_size) // 2
            start_idx = max(0, line_start - context_budget)
            end_idx = min(len(lines), line_end + context_budget)
        
        file_content = ''.join(lines[start_idx:end_idx])
        
        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
            marker = " > " if line_start <= i <= line_end else "   "
            numbered_lines.append(f"{i:4d}{marker}{line}")
        
        file_content = ''.join(numbered_lines)
        
    except Exception as e:
        file_content = f"[Error reading file: {e}]"
    
    return PromptContext(
        problem_statement=problem,
        repo_path=repo_path,
        file_path=file_path,
        file_content=file_content,
        line_start=line_start,
        line_end=line_end,
        localization_evidence=evidence,
        error_trace=error_trace
    )


if __name__ == "__main__":
    # Example usage
    ctx = PromptContext(
        problem_statement="Function returns wrong value for negative inputs",
        repo_path="/path/to/repo",
        file_path="src/calculator.py",
        file_content="""def add(a, b):
    return a + b

def subtract(a, b):
    return a - b  # Bug: should handle negatives
""",
        line_start=5,
        line_end=6,
        localization_evidence="Stack trace points to subtract function",
        error_trace="AssertionError: Expected -5, got 5"
    )
    
    system, user = PatchPromptTemplates.direct_fix_prompt(ctx)
    print("SYSTEM:", system)
    print("\nUSER:", user)
