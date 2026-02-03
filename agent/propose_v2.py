"""Unified propose module - routes through upstream intelligence.

This module integrates:
- Repair classification (taxonomy)
- Failure retrieval (memory)
- Skill routing (repo-specific)
- Planner selection (Thompson sampling)

The actual LLM patch generation is passed in as a callable.
"""
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from repair.classifier import classify_failure
from skills.router import select_skill_heads
from retrieval.failure_index import FailureIndex
from retrieval.recall import build_retrieval_context

from learning.planner_bandit import PlannerSelector, PLANNERS, register_planner
from planner.planner import generate_plan as planner_v1_generate_plan

logger = logging.getLogger(__name__)

# Register default planner
if "planner_v1" not in PLANNERS:
    register_planner("planner_v1", planner_v1_generate_plan)

# Global state
_selector = PlannerSelector()
_failure_index = FailureIndex()


@dataclass
class PatchCandidate:
    """A candidate patch from the propose pipeline."""
    patch_text: str
    summary: str
    metadata: dict[str, Any]


@dataclass
class UpstreamContext:
    """Context built by upstream intelligence modules."""
    hypotheses: list[Any]
    retrieval: dict[str, Any]
    skill_heads: list[Any]
    planner_name: str
    upstream_hints: dict[str, Any] | None = None
    file_context: dict[str, str] | None = None


def _retrieve_file_content(workspace_root: str, file_path: str, with_line_numbers: bool = True) -> str:
    """Safely read a file from the workspace, optionally with line numbers.
    
    Args:
        workspace_root: Root directory of the repository
        file_path: Relative path to the file
        with_line_numbers: If True, prepend line numbers to each line (default: True)
        
    Returns:
        File content as string, with line numbers if requested
    """
    if not workspace_root:
        return ""
    
    full_path = os.path.join(workspace_root, file_path)
    if not os.path.exists(full_path):
        # Try finding it if relative path is tricky?
        # For now, strict.
        return ""
    
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        if with_line_numbers:
            lines = content.split('\n')
            # Format: "NNNN: code" with 4-digit line numbers for alignment
            numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
            return '\n'.join(numbered_lines)
        return content
    except Exception as e:
        logger.warning("Failed to read context file %s: %s", file_path, e)
        return ""


def _scan_for_file_paths(text: str, root_dir: str = "") -> list[str]:
    """Scan text (traceback) for potential file paths."""
    # Look for patterns like 'File "foo/bar.py", line 123'
    # Or just strings ending in .py that exist on disk
    
    candidates = set()
    
    # Python traceback pattern
    tb_matches = re.findall(r'File "([^"]+)", line \d+', text)
    candidates.update(tb_matches)
    
    # Generic .py finder
    py_matches = re.findall(r'([\w/.-]+\.py)', text)
    candidates.update(py_matches)
    
    valid_files = []
    seen = set()
    
    # Validating existence if root_dir is provided helps filtering
    for path in params_priority_sort(list(candidates)): # Sort by "interestingness"?
        if path in seen: 
            continue
            
        # Normalize
        clean_path = path.strip()
        if clean_path.startswith("/"):
            # If absolute path matches workspace, relativize
            if root_dir and clean_path.startswith(root_dir):
                clean_path = os.path.relpath(clean_path, root_dir)
            else:
                # heuristic: grab end relative to generic common roots?
                # or just ignore absolute paths that aren't in workspace
                pass
        
        if root_dir and os.path.exists(os.path.join(root_dir, clean_path)):
            valid_files.append(clean_path)
            seen.add(clean_path)
            
    return valid_files[:3] # Limit to top 3


def params_priority_sort(paths: list[str]) -> list[str]:
    """Sort paths to prioritize src over tests."""
    def score(p):
        s = 0
        if "test" in p: s += 10 # Penalize tests slightly? No, tests are useful context.
        # But source is usually better for fixing.
        if "site-packages" in p: s += 100 # Penalize libs
        if "/tmp/" in p: s += 50
        return s
    return sorted(paths, key=score)


def build_upstream_context(
    task: dict[str, Any],
    last_test_output: str,
    workspace_root: str | None = None,
) -> UpstreamContext:
    """
    Build context using upstream intelligence modules.
    
    This is where all the "agent intelligence" happens:
    1. Classify the failure type
    2. Query failure index for similar past fixes
    3. Select appropriate skill heads
    4. Pick planner using Thompson sampling
    5. Retrieve relevant file content
    """
    repo = task.get("repo", "unknown")
    failing_files = task.get("failing_files", []) or []
    repo_fingerprint = task.get("repo_fingerprint", repo)

    # 1. Classify failure
    hypotheses = classify_failure(last_test_output or "", failing_files)
    logger.debug("Classified failure: %s", [h.kind for h in hypotheses[:3]])

    # 2. Query failure index
    retrieval = build_retrieval_context(repo, last_test_output or "", _failure_index)
    logger.debug("Retrieved %d similar failures", len(retrieval.get("similar_failures", [])))

    # 3. Select skill heads
    skill_heads = select_skill_heads({"repo_fingerprint": repo_fingerprint}, k=2)
    logger.debug("Selected skills: %s", [h.name for h in skill_heads])

    # 4. Pick planner (override if upstream hint present)
    upstream = task.get("_upstream", {})
    
    # 5. File Retrieval
    file_context = {}
    if workspace_root:
        # Strategy: 
        # A. Explicit hints from task
        # B. Files derived from FAIL_TO_PASS tests (highest priority - convert test path to source path)
        # C. Files mentioned in problem_statement (priority - these are likely the source files!)
        # D. Files mentioned in traceback
        
        target_files: list[str] = []
        if failing_files:
            target_files.extend(failing_files)
        
        # Derive source files from FAIL_TO_PASS test paths (highest priority)
        # e.g., "astropy/modeling/tests/test_separable.py" -> "astropy/modeling/separable.py"
        fail_to_pass = task.get("FAIL_TO_PASS", [])
        for test_path in fail_to_pass:
            # Extract file path from test spec (remove ::test_name suffix)
            test_file = test_path.split("::")[0]
            # Convert tests/test_foo.py -> foo.py or test_foo.py -> foo.py
            if "/tests/" in test_file:
                # astropy/modeling/tests/test_separable.py -> astropy/modeling/separable.py
                source_path = test_file.replace("/tests/", "/").replace("test_", "")
                if os.path.exists(os.path.join(workspace_root, source_path)):
                    target_files.insert(0, source_path)  # Highest priority
        
        # Scan problem_statement for module references (high priority)
        problem_statement = task.get("problem_statement", "")
        if problem_statement:
            # Look for import patterns like "from astropy.modeling.separable import ..."
            # or "astropy.modeling.separable.separability_matrix"
            module_patterns = re.findall(r'from\s+([\w.]+)\s+import', problem_statement)
            module_patterns += re.findall(r'([\w]+(?:\.[\w]+)+)', problem_statement)
            
            for module in module_patterns:
                # Convert module.path to file/path.py
                file_path = module.replace('.', '/') + '.py'
                if os.path.exists(os.path.join(workspace_root, file_path)):
                    target_files.insert(0, file_path)  # High priority
                    
            # Also directly scan for .py files in problem_statement
            target_files.extend(_scan_for_file_paths(problem_statement, workspace_root))
        
        if last_test_output:
            target_files.extend(_scan_for_file_paths(last_test_output, workspace_root))
            
        # Dedupe while preserving order (highest priority first)
        seen = set()
        unique_files = []
        for f in target_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        target_files = unique_files
        
        for fpath in target_files[:3]: # Cap at 3 files
            content = _retrieve_file_content(workspace_root, fpath)
            if content:
                file_context[fpath] = content
                logger.info("FILE_CONTEXT: Retrieved %s (%d lines)", fpath, content.count('\n') + 1)
                
    # Select Planner
    if upstream.get("planner"):
        planner_name = upstream["planner"]
        logger.debug("Using upstream planner override: %s", planner_name)
    else:
        planner_name = _selector.pick()
        logger.debug("Selected planner: %s", planner_name)

    return UpstreamContext(
        hypotheses=hypotheses,
        retrieval=retrieval,
        skill_heads=skill_heads,
        planner_name=planner_name,
        upstream_hints=upstream,
        file_context=file_context
    )


def propose(
    task: dict[str, Any],
    last_test_output: str,
    llm_patch_fn: Callable[[Any, dict[str, Any]], list[dict[str, Any]]],
    max_candidates: int = 6,
    workspace_root: str | None = None,
) -> list[PatchCandidate]:
    """
    Generate patch candidates using upstream intelligence.
    
    Args:
        task: Task dict with repo, description, etc.
        last_test_output: Most recent test output
        llm_patch_fn: Function(plan, context) -> list of dicts with patch_text, summary
        max_candidates: Maximum candidates to return
        workspace_root: Optional path to local repo for context retrieval
        
    Returns:
        List of PatchCandidate objects
    """
    ctx = build_upstream_context(task, last_test_output, workspace_root)

    # Get planner function
    planner_fn = PLANNERS.get(ctx.planner_name, planner_v1_generate_plan)

    # Enrich task with runtime info for LLM prompting
    enriched_task = {**task}
    enriched_task["last_test_output"] = last_test_output or ""
    
    # Generate plan (planner will store enriched task in plan.task)
    plan = planner_fn(enriched_task, ctx.retrieval)
    
    # Attach upstream context to plan metadata
    plan.metadata["repair_hypotheses"] = [h.kind for h in ctx.hypotheses]
    plan.metadata["skill_heads"] = [h.name for h in ctx.skill_heads]
    plan.metadata["retrieval"] = ctx.retrieval
    plan.metadata["files_read"] = list(ctx.file_context.keys()) if ctx.file_context else []

    # Build context dict for LLM
    llm_context = {
        "hypotheses": ctx.hypotheses,
        "retrieval": ctx.retrieval,
        "skill_heads": ctx.skill_heads,
        "planner_name": ctx.planner_name,
        "upstream_hints": ctx.upstream_hints or {},
        "file_context": ctx.file_context or {},
    }

    # Call LLM patch generator
    raw = llm_patch_fn(plan, llm_context)
    
    # Convert to PatchCandidate objects
    candidates: list[PatchCandidate] = []
    for r in raw[:max_candidates]:
        candidates.append(PatchCandidate(
            patch_text=r.get("patch_text", ""),
            summary=r.get("summary", "candidate"),
            metadata={
                "planner": ctx.planner_name,
                "hypotheses": [h.kind for h in ctx.hypotheses[:3]],
            },
        ))
    
    logger.info("Generated %d patch candidates", len(candidates))
    return candidates


def learn_update(planner_name: str, success: bool, weight: float = 1.0) -> None:
    """Update the planner bandit based on outcome."""
    _selector.update(planner_name, success=success, weight=weight)
    logger.debug("Updated planner %s: success=%s, weight=%.2f", planner_name, success, weight)


def get_propose_stats() -> dict[str, Any]:
    """Get statistics about the propose pipeline."""
    return {
        "planner_stats": _selector.get_statistics(),
        "failure_index_size": _failure_index.size(),
        "registered_planners": list(PLANNERS.keys()),
    }
