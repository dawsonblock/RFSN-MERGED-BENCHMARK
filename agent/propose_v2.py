"""Unified propose module - routes through upstream intelligence.

This module integrates:
- Repair classification (taxonomy)
- Failure retrieval (memory)
- Skill routing (repo-specific)
- Planner selection (Thompson sampling)
- Multi-layer fault localization
- Traceback signal extraction
- Import graph analysis
- Targeted test selection
- Static risk scoring
- Patch deduplication

The actual LLM patch generation is passed in as a callable.
"""
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from repair.classifier import classify_failure
from skills.router import select_skill_heads
from retrieval.failure_index import FailureIndex
from retrieval.recall import build_retrieval_context

from learning.planner_bandit import PlannerSelector, PLANNERS, register_planner
from planner.planner import generate_plan as planner_v1_generate_plan

# Multi-layer localization
try:
    from localize import localize_issue
    HAS_LOCALIZER = True
except ImportError:
    HAS_LOCALIZER = False
    localize_issue = None

# Traceback parser for structured signal extraction
from swebench_max.traceback_parser import parse_failure_signals

# Import graph for dependency analysis
from swebench_max.import_graph import build_import_graph, reverse_closure

# Targeted test selection
from swebench_max.targeted_tests_v2 import targeted_tests_v2

# Retrieval memory for episode context
from swebench_max.retrieval_memory import extract_fail_signals

# Static risk scoring
from swebench_max.static_risk import static_risk_score

# Patch deduplication
from swebench_max.dedup import PatchDeduper

# AST-based fault localization
try:
    from retrieval.ast_localization import ASTFaultLocalizer, localize_faults
    HAS_AST_LOCALIZER = True
except ImportError:
    HAS_AST_LOCALIZER = False
    ASTFaultLocalizer = None
    localize_faults = None

logger = logging.getLogger(__name__)

# Register default planner
if "planner_v1" not in PLANNERS:
    register_planner("planner_v1", planner_v1_generate_plan)

# Global state
_selector = PlannerSelector()
_failure_index = FailureIndex()
_patch_deduper = PatchDeduper()


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
    ast_fault_hints: dict[str, Any] | None = None


def _extract_error_lines(text: str, file_path: str) -> list[int]:
    """Extract line numbers mentioned for a specific file from traceback/error text.
    
    Args:
        text: Error output or traceback
        file_path: File to find line numbers for
        
    Returns:
        List of line numbers mentioned for this file
    """
    lines = []
    # Pattern: File "path/to/file.py", line 123
    pattern = rf'File "[^"]*{re.escape(os.path.basename(file_path))}", line (\d+)'
    for match in re.finditer(pattern, text):
        lines.append(int(match.group(1)))
    
    # Also check for "file.py:123:" patterns (pytest style)
    pattern2 = rf'{re.escape(os.path.basename(file_path))}:(\d+)'
    for match in re.finditer(pattern2, text):
        lines.append(int(match.group(1)))
        
    return sorted(set(lines))


def _retrieve_file_content(
    workspace_root: str, 
    file_path: str, 
    with_line_numbers: bool = True,
    window_lines: list[int] | None = None,
    window_size: int = 100,
) -> str:
    """Safely read a file from the workspace, optionally with windowing.
    
    Args:
        workspace_root: Root directory of the repository
        file_path: Relative path to the file
        with_line_numbers: If True, prepend line numbers to each line
        window_lines: Optional list of line numbers to center windows around
        window_size: Number of lines to include above and below each target line
        
    Returns:
        File content as string, with line numbers if requested.
        If window_lines provided, returns only windows around those lines.
    """
    if not workspace_root:
        return ""
    
    full_path = os.path.join(workspace_root, file_path)
    if not os.path.exists(full_path):
        return ""
    
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.read().split('\n')
        
        total_lines = len(all_lines)
        
        # If windowing is requested, extract relevant windows
        if window_lines:
            # Build set of lines to include
            include_lines = set()
            for target in window_lines:
                start = max(0, target - window_size - 1)  # 0-indexed, target is 1-indexed
                end = min(total_lines, target + window_size)
                include_lines.update(range(start, end))
            
            # Sort and identify contiguous regions
            sorted_lines = sorted(include_lines)
            segments = []
            segment_start = None
            prev_line = None
            
            for line_idx in sorted_lines:
                if segment_start is None:
                    segment_start = line_idx
                elif line_idx > prev_line + 1:
                    # Gap - end current segment, start new one
                    segments.append((segment_start, prev_line))
                    segment_start = line_idx
                prev_line = line_idx
            
            if segment_start is not None:
                segments.append((segment_start, prev_line))
            
            # Build output with segments
            result_parts = []
            for seg_start, seg_end in segments:
                if result_parts:
                    result_parts.append("\n... (lines omitted) ...\n")
                for i in range(seg_start, seg_end + 1):
                    if with_line_numbers:
                        result_parts.append(f"{i+1:4d}: {all_lines[i]}")
                    else:
                        result_parts.append(all_lines[i])
            
            if not result_parts:
                # Fallback: show first 100 lines if no specific targets found
                return _retrieve_file_content(workspace_root, file_path, with_line_numbers)
            
            return '\n'.join(result_parts)
        
        # No windowing - return full file (with optional size limit)
        if with_line_numbers:
            # Limit to 500 lines max to avoid context overflow
            if total_lines > 800:
                logger.info("File %s has %d lines, showing first 800", file_path, total_lines)
                all_lines = all_lines[:800]
                all_lines.append(f"... ({total_lines - 800} more lines omitted) ...")
            numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(all_lines)]
            return '\n'.join(numbered_lines)
        return '\n'.join(all_lines)
        
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

    # 2. Query failure index and incorporate RAG fixes from upstream
    retrieval = build_retrieval_context(repo, last_test_output or "", _failure_index)
    logger.debug("Retrieved %d similar failures", len(retrieval.get("similar_failures", [])))
    
    # Include RAG fixes passed from episode_runner
    if task.get("_similar_fixes"):
        similar_fix_context = "\n\n## Similar Past Fixes (from RAG):\n"
        for i, fix in enumerate(task["_similar_fixes"][:3], 1):
            similar_fix_context += f"\n### Fix {i} (from {fix.get('repo', 'unknown')}):\n{fix.get('patch_summary', '')[:500]}\n"
        retrieval["similar_fix_context"] = similar_fix_context
        logger.info("Added %d RAG fixes to context", len(task["_similar_fixes"]))

    # 3. Select skill heads
    skill_heads = select_skill_heads({"repo_fingerprint": repo_fingerprint}, k=2)
    logger.debug("Selected skills: %s", [h.name for h in skill_heads])

    # 4. Pick planner (override if upstream hint present)
    upstream = task.get("_upstream_hints", {})
    
    # 5. File Retrieval - Use MultiLayer Localization when available
    file_context = {}
    if workspace_root:
        target_files: list[str] = []
        problem_statement = task.get("problem_statement", "")
        fail_to_pass = task.get("FAIL_TO_PASS", [])
        
        # Parse failure signals from test output for structured extraction
        failure_signals = parse_failure_signals(last_test_output or "", workspace_root)
        logger.debug("Failure signals: %d paths, %d nodeids, %d keywords", 
                    len(failure_signals.paths), len(failure_signals.nodeids), len(failure_signals.keywords))
        
        # Priority A: Use MultiLayer Localizer if available (most sophisticated)
        if HAS_LOCALIZER and localize_issue is not None:
            try:
                hits = localize_issue(
                    problem_statement=problem_statement,
                    repo_dir=Path(workspace_root),
                    traceback=last_test_output,
                    failing_tests=fail_to_pass,
                )
                target_files = [str(h.file_path) for h in hits[:5]]
                logger.info("MultiLayer localization found %d files: %s", len(target_files), target_files[:3])
            except Exception as e:
                logger.warning("MultiLayer localization failed, falling back: %s", e)
        
        # Priority A.5: AST-based fault localization for precise function targeting
        ast_fault_hints = {}
        if HAS_AST_LOCALIZER and target_files and last_test_output:
            try:
                # Load files for AST analysis
                files_to_analyze = {}
                for fpath in target_files[:5]:
                    full_path = os.path.join(workspace_root, fpath)
                    if os.path.exists(full_path) and fpath.endswith('.py'):
                        try:
                            with open(full_path) as f:
                                files_to_analyze[fpath] = f.read()
                        except Exception:
                            pass
                
                if files_to_analyze:
                    faults = localize_faults(files_to_analyze, last_test_output, k=5)
                    for fault in faults:
                        loc = fault.code_location
                        key = f"{loc.file}:{loc.name}"
                        ast_fault_hints[key] = {
                            "file": loc.file,
                            "function": loc.name,
                            "kind": loc.kind,
                            "start_line": loc.start_line,
                            "end_line": loc.end_line,
                            "confidence": fault.confidence,
                            "reason": fault.reason,
                        }
                    if ast_fault_hints:
                        logger.info("AST localization found %d fault locations: %s", 
                                   len(ast_fault_hints), list(ast_fault_hints.keys())[:3])
            except Exception as e:
                logger.warning("AST localization failed: %s", e)
        
        # Priority B: Fallback to structured signals from traceback parser
        if not target_files and failure_signals.paths:
            target_files = list(failure_signals.paths)[:5]
            logger.debug("Using traceback parser paths: %s", target_files)
        
        # Priority C: Explicit hints from task
        if failing_files:
            target_files = list(failing_files) + target_files
        
        # Priority D: Derive source files from FAIL_TO_PASS test paths
        for test_path in fail_to_pass:
            test_file = test_path.split("::")[0]
            if "/tests/" in test_file:
                source_path = test_file.replace("/tests/", "/").replace("test_", "")
                if os.path.exists(os.path.join(workspace_root, source_path)):
                    target_files.insert(0, source_path)
        
        # Priority E: Module patterns from problem statement
        if problem_statement:
            module_patterns = re.findall(r'from\s+([\w.]+)\s+import', problem_statement)
            module_patterns += re.findall(r'([\w]+(?:\.[\w]+)+)', problem_statement)
            for module in module_patterns:
                file_path = module.replace('.', '/') + '.py'
                if os.path.exists(os.path.join(workspace_root, file_path)):
                    target_files.insert(0, file_path)
            target_files.extend(_scan_for_file_paths(problem_statement, workspace_root))
        
        # Priority F: Legacy regex from test output
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
        
        for fpath in target_files[:3]:  # Cap at 3 files
            # Extract error line numbers from test output for this file
            error_lines = _extract_error_lines(last_test_output or "", fpath)
            
            # If no specific lines found, also check problem statement
            if not error_lines and problem_statement:
                error_lines = _extract_error_lines(problem_statement, fpath)
            
            # Retrieve with windowing if we have specific lines, else full file
            if error_lines:
                content = _retrieve_file_content(
                    workspace_root, fpath, 
                    window_lines=error_lines,
                    window_size=100  # ±100 lines around each error for better context
                )
                logger.info("FILE_CONTEXT: Retrieved %s windowed around lines %s (%d lines)", 
                           fpath, error_lines[:3], content.count('\\n') + 1)
            else:
                content = _retrieve_file_content(workspace_root, fpath)
                logger.info("FILE_CONTEXT: Retrieved %s full (%d lines)", fpath, content.count('\\n') + 1)
                
            if content:
                file_context[fpath] = content
                
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
        file_context=file_context,
        ast_fault_hints=ast_fault_hints if ast_fault_hints else None,
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

    # Build import graph context for dependency analysis
    import_context = ""
    if workspace_root:
        try:
            graph = build_import_graph(workspace_root)
            changed_files = list(ctx.file_context.keys())[:5] if ctx.file_context else []
            seed_mods = set()
            for f in changed_files:
                mod = graph.file_to_mod.get(f.replace(workspace_root + "/", "").replace(workspace_root + "\\", ""))
                if mod:
                    seed_mods.add(mod)
            if seed_mods:
                dependents = reverse_closure(graph, seed_mods, depth=2, cap=50)
                import_context = f"Modules importing changed files: {', '.join(list(dependents)[:10])}"
                plan.metadata["import_graph_dependents"] = list(dependents)[:10]
        except Exception as e:
            logger.debug("Import graph analysis failed: %s", e)

    # Generate targeted test suggestions
    targeted_tests_cmds = []
    if workspace_root and last_test_output:
        try:
            from swebench_max.candidate import DiffStats
            diff_stats = DiffStats(
                files_changed=len(ctx.file_context.keys()) if ctx.file_context else 0,
                lines_changed=0,
                paths=list(ctx.file_context.keys())[:10] if ctx.file_context else [],
            )
            targeted_tests_cmds = targeted_tests_v2(
                diff=diff_stats,
                repo_root=workspace_root,
                failures_text=last_test_output,
                limit=5,
            )
            if targeted_tests_cmds:
                plan.metadata["targeted_tests"] = targeted_tests_cmds[:5]
                logger.debug("Suggested targeted tests: %s", targeted_tests_cmds[:3])
        except Exception as e:
            logger.debug("Targeted test selection failed: %s", e)

    # Add episode retrieval context for cross-task learning
    fail_signals = []
    if workspace_root:
        try:
            instance_id = task.get("instance_id", "")
            log_dir = os.path.join(workspace_root, "..", "logs")
            if os.path.isdir(log_dir) and instance_id:
                fail_signals = extract_fail_signals(log_dir, instance_id, limit=3)
                if fail_signals:
                    plan.metadata["past_failures"] = fail_signals
                    logger.debug("Retrieved %d past failure signals", len(fail_signals))
        except Exception as e:
            logger.debug("Episode retrieval failed: %s", e)

    # Build context dict for LLM
    llm_context = {
        "hypotheses": ctx.hypotheses,
        "retrieval": ctx.retrieval,
        "skill_heads": ctx.skill_heads,
        "planner_name": ctx.planner_name,
        "upstream_hints": ctx.upstream_hints or {},
        "file_context": ctx.file_context or {},
        "import_context": import_context,
        "targeted_tests": targeted_tests_cmds[:3],
        "past_failures": fail_signals[:2] if fail_signals else [],
    }

    # Call LLM patch generator
    raw = llm_patch_fn(plan, llm_context)
    
    # Convert to PatchCandidate objects with deduplication and risk scoring
    candidates: list[PatchCandidate] = []
    forbid_prefixes = task.get("forbid_paths_prefix", [])
    
    for r in raw[:max_candidates * 2]:  # Process more to account for deduplication
        patch_text = r.get("patch_text", "")
        if not patch_text:
            continue
            
        # Deduplicate patches
        if not _patch_deduper.add(patch_text):
            logger.debug("Skipping duplicate patch")
            continue
        
        # Calculate static risk score
        risk = static_risk_score(patch_text, forbid_prefixes)
        
        candidates.append(PatchCandidate(
            patch_text=patch_text,
            summary=r.get("summary", "candidate"),
            metadata={
                "planner": ctx.planner_name,
                "hypotheses": [h.kind for h in ctx.hypotheses[:3]],
                "static_risk": risk,
                "variant": r.get("variant", "default"),
            },
        ))
        
        if len(candidates) >= max_candidates:
            break
    
    # Sort candidates by static risk (lower is better, so higher scores first)
    candidates.sort(key=lambda c: c.metadata.get("static_risk", -999), reverse=True)
    
    logger.info("Generated %d unique patch candidates (after dedup)", len(candidates))
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
