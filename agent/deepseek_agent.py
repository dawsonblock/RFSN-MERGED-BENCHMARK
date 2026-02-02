"""SWE-bench agent functions wired to DeepSeek R1.

This module provides the three core functions needed by the agent loop:
- propose_fn: Uses DeepSeek R1 to generate proposals
- gate_fn: Validates proposals against profile constraints
- exec_fn: Executes proposals (apply patches, run tests, etc.)
"""

from __future__ import annotations

import re
import subprocess  # legacy; avoid direct use (see _run_argv)
import tempfile
from pathlib import Path

from agent.types import (
    AgentState,
    Proposal,
    GateDecision,
    ExecResult,
    Phase,
)
from agent.profiles import Profile

# Try to import the DeepSeek client
try:
    from rfsn_controller.llm.deepseek import call_model
    HAS_DEEPSEEK = True
except ImportError:
    HAS_DEEPSEEK = False
    def call_model(model_input: str, temperature: float = 0.0) -> dict:
        """Fallback when DeepSeek is not available."""
        return {
            "mode": "tool_request",
            "requests": [],
            "why": "DeepSeek client not available",
        }

try:
    from rfsn_controller.structured_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


# =============================================================================
# HARDENED SUBPROCESS WRAPPER (GATE HARDENING)
# =============================================================================

def _run_argv(workdir: Path, argv: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    """Run a command with strict argv-only execution.

    This routes through rfsn_controller.exec_utils.safe_run when available,
    which enforces:
      - no shell wrappers (sh -c, bash -c, ...)
      - allowlist checks
      - sanitized env

    Returns:
      (returncode, stdout, stderr)
    """
    try:
        from rfsn_controller.exec_utils import safe_run
        r = safe_run(argv=argv, cwd=str(workdir), timeout_sec=timeout_sec)
        return (0 if r.ok else int(r.exit_code), r.stdout or "", r.stderr or "")
    except Exception as e:
        # Fail closed: do not fall back to shell execution.
        return (1, "", f"safe_run failed: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _clean_search_term(term: str) -> str:
    """Clean a search term by removing punctuation and formatting characters."""
    # Remove markdown formatting
    term = term.replace("`", "").replace("*", "").replace("_", " ").replace("__", " ")
    # Remove common punctuation
    term = re.sub(r"['\"`.,;:!?\(\)\[\]\{\}<>]", "", term)
    # Clean up whitespace
    term = term.strip()
    return term


def _extract_code_identifiers(text: str) -> list[str]:
    """Extract likely code identifiers from problem text."""
    identifiers = []
    
    # Find function/method names (word_word pattern)
    snake_case = re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text)
    identifiers.extend(snake_case)
    
    # Find class names (CamelCase)
    camel_case = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
    identifiers.extend(camel_case)
    
    # Find module paths (word.word.word)
    module_paths = re.findall(r'\b([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+)\b', text)
    identifiers.extend(module_paths)
    
    # Find backtick-wrapped code
    backtick_code = re.findall(r'`([^`]+)`', text)
    for code in backtick_code:
        clean = _clean_search_term(code)
        if len(clean) > 3 and clean.replace("_", "").isalnum():
            identifiers.append(clean)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for ident in identifiers:
        if ident not in seen and len(ident) > 3:
            seen.add(ident)
            unique.append(ident)
    
    return unique[:20]  # Limit to 20


# =============================================================================
# PROPOSE FUNCTION - Uses DeepSeek R1 to generate proposals
# =============================================================================

def propose(profile: Profile, state: AgentState) -> Proposal:
    """Generate a proposal using DeepSeek R1."""
    prompt = _build_prompt(profile, state)
    
    # Temperature escalation: increase creativity when stuck
    base_temp = 0.0
    reject_count = state.notes.get("consecutive_rejects", 0)
    phase_attempts = state.notes.get(f"phase_{state.phase.value}_attempts", 0)
    
    # Escalate temperature when struggling
    if reject_count >= 2:
        base_temp = 0.3  # More creative after multiple rejects
    elif phase_attempts >= 3:
        base_temp = 0.2  # Slightly more creative when stuck in phase
    
    logger.debug(
        "Calling DeepSeek R1",
        phase=state.phase.value, 
        prompt_len=len(prompt),
        temperature=base_temp,
        reject_count=reject_count,
    )
    
    try:
        response = call_model(prompt, temperature=base_temp)
    except Exception as e:
        logger.error("DeepSeek call failed", error=str(e))
        return _fallback_proposal(state, str(e))
    
    return _parse_response_to_proposal(response, state)


def _fallback_proposal(state: AgentState, error: str) -> Proposal:
    """Generate a fallback proposal based on current phase."""
    problem = state.notes.get("problem_statement", "")
    
    # In LOCALIZE, try to search for keywords from problem statement
    if state.phase == Phase.LOCALIZE:
        # Extract potential search terms from problem
        keywords = [w for w in problem.split() if len(w) > 5 and w.isalpha()][:3]
        if keywords:
            return Proposal(
                kind="search",
                rationale=f"Fallback search for: {keywords[0]}",
                inputs={"query": keywords[0]},
                evidence=[],
            )
    
    return Proposal(
        kind="inspect",
        rationale=f"LLM call failed: {error}",
        inputs={"query": "problem_statement"},
        evidence=[],
    )

def _build_prompt(profile: Profile, state: AgentState) -> str:
    """Build a comprehensive prompt with history for DeepSeek."""
    parts = []
    
    # Task context - most important
    problem = state.notes.get("problem_statement", "")
    parts.append(f"# 🎯 TASK\n{problem}")
    
    # Current phase with detailed instructions - up front so LLM knows what to do
    parts.append(f"\n# 📋 CURRENT PHASE: {state.phase.value}")
    parts.append(_get_phase_instruction(state.phase))
    
    # For PATCH_CANDIDATES, explicitly show which file to edit
    if state.phase == Phase.PATCH_CANDIDATES:
        last_contents = state.notes.get("last_file_contents", {})
        if last_contents:
            source_files = [f for f in last_contents.keys() if not '/test' in f and not 'test_' in f]
            if source_files:
                main_file = source_files[0]
                parts.append(f"\n# 🎯 FILE TO EDIT: {main_file}")
                parts.append("Generate a patch for this file based on the content shown below.")
    
    # File contents from last read - SHOW PROMINENTLY WITH LINE NUMBERS
    last_contents = state.notes.get("last_file_contents", {})
    if last_contents:
        parts.append("\n# 📄 FILE CONTENTS (use exact text for your diff)")
        for fname, content in list(last_contents.items())[:3]:
            # Skip test files in PATCH_CANDIDATES to reduce confusion
            if state.phase == Phase.PATCH_CANDIDATES and ('test_' in fname or '/test' in fname):
                parts.append(f"\n## ⚠️ SKIP: {fname} (test file - do not edit)")
                continue
            # Add line numbers to help with diff generation
            lines = content.split("\n")[:150]  # Limit to 150 lines
            numbered_lines = [f"{i+1:4}: {line}" for i, line in enumerate(lines)]
            numbered_content = "\n".join(numbered_lines)
            parts.append(f"\n## ✅ {fname}\n```\n{numbered_content}\n```")
    
    # Budget status - show urgency
    remaining_rounds = profile.max_rounds - state.budget.round_idx
    parts.append(f"\n# ⏱️ BUDGET: {remaining_rounds} rounds remaining")
    if remaining_rounds <= 2:
        parts.append("⚠️ LOW BUDGET - Generate a patch NOW!")
    
    # Localization hits - files that likely need changes
    if state.localization_hits:
        source_hits = [h for h in state.localization_hits if not 'test' in h.get('file', '').lower()]
        if source_hits:
            parts.append("\n# 📍 LIKELY BUG LOCATIONS")
            for hit in source_hits[:5]:
                parts.append(f"- {hit.get('file', 'unknown')}: {hit.get('reason', '')}")
    
    # Add history of what was already done (compact)
    if "action_history" not in state.notes:
        state.notes["action_history"] = []
    
    history = state.notes.get("action_history", [])
    if history:
        parts.append(f"\n# 📜 HISTORY (last {min(len(history), 5)} actions)")
        for h in history[-5:]:
            parts.append(f"- {h}")
    
    # Last failures - important for diagnosis
    if state.last_failures:
        parts.append("\n# ❌ LAST TEST FAILURES")
        for f in state.last_failures[:3]:
            parts.append(f"- {f.nodeid}: {f.message[:100]}")
    
    # Last gate rejection
    if "last_gate_reject" in state.notes:
        parts.append(f"\n# ⛔ LAST REJECTION: {state.notes['last_gate_reject']}")
    
    # Force source edit guidance after multiple test file rejects
    if state.notes.get("force_source_edit"):
        parts.append("\n# 🚨 CRITICAL: Your last patches tried to edit TEST files!")
        parts.append("You MUST edit SOURCE code files only. Look for:")
        parts.append("- Files in src/, lib/, or the main package (not tests/)")
        parts.append("- The file containing the bug, not the file testing it")
        parts.append("- Implementation code with the broken logic")
    
    # Outcome learning hints - show similar past patches
    if "last_diff" in state.notes:
        try:
            from agent.outcome_learning import predict_patch_success, get_outcome_learner
            last_diff = state.notes["last_diff"]
            pred = predict_patch_success(last_diff)
            if pred < 0.3:
                parts.append("\n# ⚠️ WARNING: Similar patches have low success rates")
                parts.append("Consider a different approach to this fix.")
            
            # Show similar patches if available
            learner = get_outcome_learner()
            similar = learner.get_similar_patches(last_diff, limit=2)
            if similar:
                parts.append("\n# 📊 SIMILAR PAST PATCHES:")
                for patch in similar:
                    status = "✅ SUCCESS" if patch["success"] else "❌ FAILED"
                    parts.append(f"- {status}: {patch['patterns'][:3]}")
        except Exception:
            pass  # Outcome learning not available
    
    return "\n".join(parts)


def _get_phase_instruction(phase: Phase) -> str:
    """Get detailed instruction for current phase."""
    instructions = {
        Phase.INGEST: """
## PHASE: INGEST - Understand the Problem

Read the problem statement and identify:
1. What is broken or needs to change?
2. Which file(s) likely contain the bug?
3. What behavior should change?

OUTPUT: {"mode": "tool_request", "requests": [{"tool": "sandbox.read_file", "args": {"path": "<most_likely_source_file>"}}], "why": "Reading source to understand current behavior"}
""",
        Phase.LOCALIZE: """
## PHASE: LOCALIZE - Find the Bug Location

You've read some code. Now pinpoint the EXACT location of the bug.

1. Search for function/class names mentioned in the problem
2. Search for error messages or specific patterns
3. Read the actual source file (NOT test files)

OUTPUT: {"mode": "tool_request", "requests": [{"tool": "sandbox.grep", "args": {"pattern": "<function_or_class_name>", "path": "."}}], "why": "Finding exact location of bug"}

IMPORTANT: Focus on SOURCE files in src/, lib/, or the main package directory. Avoid test/ directories.
""",
        Phase.PLAN: """
## PHASE: PLAN - Design the Fix

Based on what you've read, plan the minimal fix:

1. What specific line(s) need to change?
2. What is the old (buggy) code?
3. What should the new (fixed) code be?

If you need more info, read another file. Otherwise, proceed to generate a patch.

OUTPUT: Continue reading OR proceed to patch generation
""",
        Phase.PATCH_CANDIDATES: """
## PHASE: PATCH_CANDIDATES - Generate the Fix

⚠️ YOU MUST OUTPUT A PATCH NOW. No more reading files.

OUTPUT THIS EXACT JSON FORMAT:
```json
{
  "mode": "patch",
  "diff": "--- a/path/to/file.py\\n+++ b/path/to/file.py\\n@@ -LINE,COUNT +LINE,COUNT @@\\n context\\n-old_line\\n+new_line\\n context",
  "why": "Brief explanation of fix"
}
```

ABSOLUTE RULES:
1. ⛔ NEVER edit test files - only edit SOURCE code
2. ✅ Edit the file where the bug lives (from your earlier reading)
3. ✅ Use EXACT text from the file you read (copy-paste the line)
4. ✅ The - line (removed) MUST be different from + line (added)
5. ✅ Include 1-2 context lines before/after your change

DIFF FORMAT:
- Start with: --- a/filepath
- Then: +++ b/filepath
- Then: @@ -startline,count +startline,count @@
- Context lines: no prefix
- Removed lines: start with -
- Added lines: start with +

EXAMPLE (fixing a comparison bug):
{"mode": "patch", "diff": "--- a/mylib/core.py\\n+++ b/mylib/core.py\\n@@ -42,3 +42,3 @@\\n     def compare(self, x, y):\\n-        return x == y\\n+        return x is y\\n", "why": "Changed equality to identity comparison as required"}
""",
        Phase.TEST_STAGE: """
## PHASE: TEST_STAGE - Verify the Fix

Run the relevant tests to confirm your fix works.

OUTPUT: {"mode": "tool_request", "requests": [{"tool": "sandbox.run_command", "args": {"cmd": ["pytest", "path/to/test.py", "-v"]}}], "why": "Verifying fix passes tests"}

If no specific test is mentioned, run: pytest -x --tb=short
""",
        Phase.DIAGNOSE: """
## PHASE: DIAGNOSE - Analyze Test Failures

Tests failed. Analyze WHY:

1. Read the test that failed
2. Compare expected vs actual behavior
3. Check if your patch was applied correctly
4. Identify what needs to change

OUTPUT: Read the failing test file or the patched source file
""",
        Phase.MINIMIZE: """
## PHASE: MINIMIZE - Clean Up

Ensure your fix is minimal:
1. Remove any unnecessary changes
2. Verify tests still pass
3. No formatting-only changes

OUTPUT: Final verification or adjusted patch
""",
        Phase.FINALIZE: """
## PHASE: FINALIZE - Complete

Tests pass! Finalize the solution.

OUTPUT: {"mode": "feature_summary", "summary": "Fixed [problem] by [change made]", "completion_status": "complete"}
""",
    }
    return instructions.get(phase, "Proceed with the task.")


def _parse_response_to_proposal(response: dict, state: AgentState) -> Proposal:
    """Parse DeepSeek JSON response into a Proposal."""
    mode = response.get("mode", "tool_request")
    why = response.get("why", "")
    
    # Track this action in history
    if "action_history" not in state.notes:
        state.notes["action_history"] = []
    
    if mode == "patch":
        diff = response.get("diff", "")
        files = _extract_files_from_diff(diff)
        state.notes["action_history"].append(f"Generated patch for {files}")
        return Proposal(
            kind="edit",
            rationale=why or "Generated patch",
            inputs={"diff": diff, "files": files},
            evidence=[],
        )
    
    elif mode == "tool_request":
        requests = response.get("requests", [])
        if not requests:
            return _infer_proposal_from_phase(state, why)
        
        # Parse all requests
        first_req = requests[0]
        tool = first_req.get("tool", "")
        args = first_req.get("args", {})
        
        # Determine proposal kind and inputs
        if any(kw in tool for kw in ["grep", "search", "find", "rg"]):
            query = args.get("query", args.get("pattern", ""))
            state.notes["action_history"].append(f"Search: {query}")
            return Proposal(
                kind="search",
                rationale=why or f"Search for: {query}",
                inputs={"query": query},
                evidence=[],
            )
        
        elif any(kw in tool for kw in ["read", "cat", "view"]):
            path = args.get("path", args.get("file", ""))
            # Avoid re-reading the same file
            files_read = state.notes.get("files_read", [])
            if path in files_read:
                # Force a search instead
                return _infer_proposal_from_phase(state, f"Already read {path}, trying search")
            state.notes["action_history"].append(f"Read: {path}")
            return Proposal(
                kind="inspect",
                rationale=why or f"Read file: {path}",
                inputs={"files": [path]},
                evidence=[],
            )
        
        elif any(kw in tool for kw in ["test", "pytest", "run"]):
            cmd = args.get("command", args.get("cmd", "pytest"))
            state.notes["action_history"].append(f"Run: {cmd}")
            return Proposal(
                kind="run_tests",
                rationale=why or f"Run tests: {cmd}",
                inputs={"command": cmd},
                evidence=[],
            )
        
        else:
            # Unknown tool - try to infer from args
            if "path" in args:
                return Proposal(
                    kind="inspect",
                    rationale=why or "Read file",
                    inputs={"files": [args["path"]]},
                    evidence=[],
                )
            return _infer_proposal_from_phase(state, why)
    
    elif mode == "feature_summary":
        state.notes["action_history"].append("Finalize")
        return Proposal(
            kind="finalize",
            rationale=why or "Task complete",
            inputs={
                "summary": response.get("summary", ""),
                "status": response.get("completion_status", "complete"),
            },
            evidence=[],
        )
    
    return _infer_proposal_from_phase(state, why)


def _infer_proposal_from_phase(state: AgentState, why: str) -> Proposal:
    """Infer a sensible proposal based on current phase when parsing fails."""
    problem = state.notes.get("problem_statement", "")
    files_read = state.notes.get("files_read", [])
    history = state.notes.get("action_history", [])
    last_contents = state.notes.get("last_file_contents", {})
    
    # Use regex-based extraction for clean code identifiers
    code_terms = _extract_code_identifiers(problem)
    
    # Check if we've already searched these terms
    searched_terms = [h.split(": ")[-1] for h in history if "search" in h.lower()]
    unsearched_terms = [t for t in code_terms if t not in searched_terms]
    
    # CRITICAL: In PATCH_CANDIDATES phase with file contents - force edit proposal
    if state.phase == Phase.PATCH_CANDIDATES and last_contents and files_read:
        # We have read files and have content - generate a placeholder edit
        first_file = list(last_contents.keys())[0]
        state.notes["action_history"].append(f"Forcing edit for: {first_file}")
        return Proposal(
            kind="edit",
            rationale="Generating patch based on analyzed file contents",
            inputs={
                "files": [first_file],
                "diff": "",  # LLM should have generated this, but fallback is empty
            },
            evidence=[],
        )
    
    # In LOCALIZE, PLAN phases - try searching if we have clean terms
    if state.phase in [Phase.LOCALIZE, Phase.PLAN] and unsearched_terms:
        term = unsearched_terms[0]
        state.notes["action_history"].append(f"Inferred search: {term}")
        return Proposal(
            kind="search",
            rationale=why or f"Search for code term: {term}",
            inputs={"query": term},
            evidence=[],
        )
    
    # If we have localization hits, read the most recent unread file
    if state.localization_hits:
        for hit in state.localization_hits:
            fname = hit.get("file", "")
            if fname and fname not in files_read:
                state.notes["action_history"].append(f"Inferred read: {fname}")
                return Proposal(
                    kind="inspect",
                    rationale=why or f"Read localized file: {fname}",
                    inputs={"files": [fname]},
                    evidence=[],
                )
    
    # Try fallback search terms if no code identifiers found
    if state.phase in [Phase.LOCALIZE, Phase.PLAN, Phase.PATCH_CANDIDATES]:
        # Search for common patterns
        fallback_terms = ["def ", "class ", "import "]
        for term in fallback_terms:
            if term not in searched_terms:
                state.notes["action_history"].append(f"Fallback search: {term}")
                return Proposal(
                    kind="search",
                    rationale=f"Searching for {term.strip()} patterns",
                    inputs={"query": term},
                    evidence=[],
                )
    
    # Only fall back to problem_statement once
    ps_reads = sum(1 for h in history if "problem_statement" in h.lower())
    if ps_reads < 1:
        state.notes["action_history"].append("Read problem_statement")
        return Proposal(
            kind="inspect",
            rationale=why or "Review problem statement",
            inputs={"query": "problem_statement"},
            evidence=[],
        )
    
    # Force finalize if we've exhausted all options
    if state.phase in [Phase.PATCH_CANDIDATES, Phase.PLAN]:
        state.notes["action_history"].append("Force finalize - no more actions")
        return Proposal(
            kind="finalize",
            rationale="Exhausted search options",
            inputs={"summary": "Could not generate a valid patch", "status": "incomplete"},
            evidence=[],
        )
    
    # Generic search as absolute last resort
    state.notes["action_history"].append("Generic search: error")
    return Proposal(
        kind="search",
        rationale="Searching for error patterns",
        inputs={"query": "error"},
        evidence=[],
    )


def _extract_files_from_diff(diff: str) -> list[str]:
    """Extract file paths from a unified diff."""
    files = []
    for line in diff.split("\n"):
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                files.append(path)
    return list(set(files))


# =============================================================================
# GATE FUNCTION - Validates proposals against constraints
# =============================================================================

def gate(profile: Profile, state: AgentState, proposal: Proposal) -> GateDecision:
    """
    Validate a proposal against profile constraints using the kernel's PlanGate.

    This implementation delegates all policy enforcement to the single source of
    truth gate defined in ``rfsn_controller/gates/plan_gate.py`` via the
    ``GateAdapter``. It converts the agent's state and proposal into a simple
    dictionary snapshot for the adapter, then forwards the decision back into
    the agent's ``GateDecision`` contract.
    """
    # Lazy imports to avoid circular dependency and heavy type imports
    from typing import Any, Dict
    from agent.gate_adapter import get_gate_adapter

    # Build a minimal snapshot of the agent state expected by the adapter.
    # Only include the fields that PlanGate needs to make a decision.
    state_snapshot: Dict[str, Any] = {
        "attempt": state.attempt,
        "phase": state.phase.value,
        "budget": {
            "test_runs": state.budget.test_runs,
            "patch_attempts": state.budget.patch_attempts,
        },
        # Copy notes to allow policies to inspect custom counters or flags.
        "notes": dict(state.notes),
    }

    # Map the proposal into the adapter format.
    prop_dict: Dict[str, Any] = {
        "type": proposal.kind,
        "summary": proposal.rationale,
        "inputs": dict(proposal.inputs),
    }

    # Extract patch text if present. The adapter expects unified diff in
    # ``patch_text`` for patch proposals.
    diff = proposal.inputs.get("diff") or proposal.inputs.get("patch")
    if isinstance(diff, str) and diff:
        prop_dict["patch_text"] = diff

    adapter = get_gate_adapter()
    decision = adapter.decide(state_snapshot=state_snapshot, proposal=prop_dict)

    # Convert the adapter's GateDecision into the agent's GateDecision.
    return GateDecision(
        accept=bool(decision.allowed),
        reason=str(decision.reason),
        constraints=dict(decision.metadata) if hasattr(decision, "metadata") else {},
    )


def _allowed_kinds_for_phase(phase: Phase) -> list[str]:
    """Get allowed proposal kinds for a given phase."""
    return {
        Phase.INGEST: ["inspect", "search"],
        Phase.LOCALIZE: ["inspect", "search"],
        Phase.PLAN: ["inspect", "search", "edit"],  # Allow edit from PLAN
        Phase.PATCH_CANDIDATES: ["edit", "inspect", "search"],
        Phase.TEST_STAGE: ["run_tests", "inspect"],
        Phase.DIAGNOSE: ["inspect", "search", "edit"],
        Phase.MINIMIZE: ["edit", "inspect", "run_tests"],
        Phase.FINALIZE: ["finalize", "run_tests"],
        Phase.DONE: [],
    }.get(phase, ["inspect"])


# =============================================================================
# EXEC FUNCTION - Executes proposals
# =============================================================================

def execute(profile: Profile, state: AgentState, proposal: Proposal) -> ExecResult:
    """Execute a proposal."""
    logger.info("Executing proposal", kind=proposal.kind)
    
    if proposal.kind == "inspect":
        return _exec_inspect(state, proposal)
    elif proposal.kind == "search":
        return _exec_search(state, proposal)
    elif proposal.kind == "edit":
        return _exec_edit(state, proposal)
    elif proposal.kind == "run_tests":
        return _exec_run_tests(state, proposal)
    elif proposal.kind == "finalize":
        return _exec_finalize(state, proposal)
    else:
        return ExecResult(status="fail", summary=f"Unknown proposal kind: {proposal.kind}")


def _exec_inspect(state: AgentState, proposal: Proposal) -> ExecResult:
    """Execute an inspect proposal."""
    workdir = Path(state.repo.workdir)
    
    files = proposal.inputs.get("files", [])
    query = proposal.inputs.get("query", "")
    
    if query == "problem_statement":
        content = state.notes.get("problem_statement", "No problem statement available")
        return ExecResult(
            status="ok",
            summary=f"Problem statement: {content[:200]}...",
            artifacts=[],
            metrics={"content": content},
        )
    
    if files:
        contents = {}
        files_read = state.notes.get("files_read", [])
        
        for f in files:
            path = workdir / f
            if path.exists():
                try:
                    file_content = path.read_text()[:8000]
                    contents[f] = file_content
                    if f not in files_read:
                        files_read.append(f)
                except Exception as e:
                    contents[f] = f"Error reading: {e}"
            else:
                contents[f] = "File not found"
        
        state.notes["files_read"] = files_read
        state.notes["last_file_contents"] = contents
        
        # Add to localization hits if file contains useful info
        for fname in contents:
            if contents[fname] != "File not found":
                state.localization_hits.append({
                    "file": fname,
                    "reason": "file read",
                    "type": "read",
                })
        
        return ExecResult(
            status="ok",
            summary=f"Read {len(contents)} files: {list(contents.keys())}",
            artifacts=list(contents.keys()),
            metrics={"contents": contents},
        )
    
    return ExecResult(status="ok", summary="No files to inspect")


def _exec_search(state: AgentState, proposal: Proposal) -> ExecResult:
    """Execute a search proposal using ripgrep, grep, or Python fallback."""
    workdir = Path(state.repo.workdir)
    query = proposal.inputs.get("query", "")
    
    if not query:
        return ExecResult(status="fail", summary="No search query provided")
    
    files: list[str] = []

    # HARDENING: avoid external grep/rg subprocess calls here.
    # Use a Python search to keep the execution surface small and deterministic.
    try:
        for root, dirs, filenames in workdir.walk():
            # Skip hidden and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in [
                "__pycache__", "node_modules", "venv", ".git", "build", "dist"
            ]]

            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = Path(root) / fname
                try:
                    content = fpath.read_text(errors="ignore")
                except Exception:
                    continue
                if query in content:
                    try:
                        rel_path = str(fpath.relative_to(workdir))
                    except Exception:
                        rel_path = str(fpath)
                    files.append(rel_path)
                    if len(files) >= 20:
                        break

            if len(files) >= 20:
                break
    except Exception as e:
        return ExecResult(status="fail", summary=f"Python search failed: {e}")
    
    # Update localization hits
    for f in files:
        state.localization_hits.append({
            "file": f,
            "reason": f"match for '{query}'",
            "type": "search",
        })
    
    if files:
        return ExecResult(
            status="ok",
            summary=f"Found {len(files)} files matching '{query}': {files[:5]}",
            artifacts=files,
            metrics={"query": query, "matches": files},
        )
    else:
        return ExecResult(
            status="ok",
            summary=f"No files found matching '{query}'",
            artifacts=[],
            metrics={"query": query, "matches": []},
        )


def _exec_edit(state: AgentState, proposal: Proposal) -> ExecResult:
    """Execute an edit proposal by applying a patch."""
    workdir = Path(state.repo.workdir)
    diff = proposal.inputs.get("diff", "")
    
    if not diff:
        return ExecResult(status="fail", summary="No diff provided")
    
    # Validate patch before attempting to apply
    is_valid, error = _validate_patch(diff)
    if not is_valid:
        logger.warning("Invalid patch detected", error=error)
        return ExecResult(
            status="fail",
            summary=f"Invalid patch: {error}",
            metrics={"validation_error": error, "diff_preview": diff[:500]},
        )
    
    # Validate patch produces valid Python syntax
    is_valid, error = _validate_patch_syntax(diff, workdir)
    if not is_valid:
        logger.warning("Patch syntax validation failed", error=error)
        return ExecResult(
            status="fail",
            summary=f"Syntax error: {error}",
            metrics={"syntax_error": error, "diff_preview": diff[:500]},
        )
    
    # Try to repair common patch issues
    diff = _repair_patch(diff)
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(diff)
            patch_path = f.name
        
        # Check first (argv-only)
        rc, out, err = _run_argv(workdir, ["git", "apply", "--check", patch_path], timeout_sec=60)
        
        if rc != 0:
            # Log the failed patch for debugging
            logger.warning(
                "Patch check failed",
                error=(err or "")[:500],
                diff_preview=diff[:500],
            )
            
            # Try to apply as a direct file edit if patch fails
            edit_result = _try_direct_edit(state, proposal, diff)
            if edit_result:
                return edit_result
            
            # Try structured SEARCH/REPLACE approach
            structured_result = _try_structured_edit(state, proposal, diff)
            if structured_result:
                return structured_result
            
            return ExecResult(
                status="fail",
                summary=f"Patch check failed: {(err or '')[:200]}",
                metrics={"stderr": err, "diff": diff[:1000]},
            )
        
        # Apply with --3way for more lenient patching
        rc, out, err = _run_argv(workdir, ["git", "apply", "--3way", patch_path], timeout_sec=60)
        
        if rc == 0:
            files = proposal.inputs.get("files", [])
            state.budget.patch_attempts += 1
            return ExecResult(
                status="ok",
                summary=f"Patch applied successfully ({len(files)} files)",
                artifacts=files,
            )
        else:
            return ExecResult(status="fail", summary=f"Patch apply failed: {(err or '')[:200]}")
    except Exception as e:
        return ExecResult(status="fail", summary=f"Edit failed: {e}")


def _validate_patch(diff: str) -> tuple[bool, str]:
    """Validate that a patch contains actual changes.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not diff or not diff.strip():
        return False, "Empty diff"
    
    lines = diff.split("\n")
    removals = []
    additions = []
    
    for line in lines:
        # Skip headers and context
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("diff --git"):
            continue
        
        # Collect actual changes
        if line.startswith("-") and not line.startswith("---"):
            removals.append(line[1:])  # Strip the - prefix
        elif line.startswith("+") and not line.startswith("+++"):
            additions.append(line[1:])  # Strip the + prefix
    
    if not removals and not additions:
        return False, "No changes in diff (no + or - lines)"
    
    # Check for no-op: all removals match all additions
    if removals == additions:
        return False, "No-op patch: removed lines identical to added lines"
    
    # Check for minimal change (at least one actual difference)
    has_real_change = False
    for i, (rem, add) in enumerate(zip(removals, additions)):
        if rem.strip() != add.strip():
            has_real_change = True
            break
    
    # If lengths differ, there's definitely a change
    if len(removals) != len(additions):
        has_real_change = True
    
    if not has_real_change and removals and additions:
        return False, "No meaningful changes (only whitespace differences)"
    
    return True, ""


def _validate_patch_syntax(diff: str, workdir: Path) -> tuple[bool, str]:
    """Validate that Python patches produce valid syntax.
    
    Applies the patch virtually and parses the result as AST.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    import ast
    
    # Extract files from diff
    files = _extract_files_from_diff(diff)
    if not files:
        return True, ""  # Can't validate without file info
    
    for file_path in files:
        # Only validate Python files
        if not file_path.endswith(".py"):
            continue
        
        full_path = workdir / file_path
        if not full_path.exists():
            continue
        
        try:
            original = full_path.read_text()
            patched = _simulate_patch(original, diff, file_path)
            
            if patched is None:
                continue  # Could not simulate patch
            
            # Try to parse as Python AST
            ast.parse(patched)
        except SyntaxError as e:
            return False, f"Patch creates syntax error in {file_path}: {e.msg} at line {e.lineno}"
        except Exception:
            pass  # Non-Python file or other issue, skip validation
    
    return True, ""


def _simulate_patch(original: str, diff: str, file_path: str) -> str | None:
    """Simulate applying a patch to content without actually writing.
    
    Returns:
        Patched content or None if simulation failed
    """
    lines = original.splitlines(keepends=True)
    diff_lines = diff.split("\n")
    
    # Find hunks for this file
    in_file = False
    hunks = []
    current_hunk = None
    
    for line in diff_lines:
        if line.startswith("--- a/") or line.startswith("--- "):
            in_file = file_path in line
        elif line.startswith("+++ b/") or line.startswith("+++ "):
            in_file = file_path in line
        elif line.startswith("@@") and in_file:
            # Parse hunk header: @@ -start,len +start,len @@
            import re
            match = re.match(r"@@ -(\d+)", line)
            if match:
                start_line = int(match.group(1))
                current_hunk = {"start": start_line - 1, "removals": [], "additions": []}
                hunks.append(current_hunk)
        elif current_hunk is not None and in_file:
            if line.startswith("-") and not line.startswith("---"):
                current_hunk["removals"].append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                current_hunk["additions"].append(line[1:])
            elif line.startswith(" "):
                pass  # Context line
    
    if not hunks:
        return None
    
    # Apply hunks (simplified - just replace removals with additions)
    result_lines = list(lines)
    offset = 0
    
    for hunk in hunks:
        start = hunk["start"] + offset
        removals = len(hunk["removals"])
        additions = hunk["additions"]
        
        # Remove old lines and insert new ones
        del result_lines[start:start + removals]
        for i, add_line in enumerate(additions):
            if not add_line.endswith("\n"):
                add_line += "\n"
            result_lines.insert(start + i, add_line)
        
        offset += len(additions) - removals
    
    return "".join(result_lines)


def _try_structured_edit(state: AgentState, proposal: Proposal, diff: str) -> ExecResult | None:
    """Try to apply changes using SEARCH/REPLACE blocks extracted from diff.
    
    This is a more robust approach when unified diffs fail.
    """
    workdir = Path(state.repo.workdir)
    files = proposal.inputs.get("files", [])
    
    if not files:
        # Try to extract from diff
        files = _extract_files_from_diff(diff)
    
    if not files:
        return None
    
    # Parse diff to extract old/new code blocks
    lines = diff.split("\n")
    current_file = None
    removals = []
    additions = []
    changes_made = False
    
    for line in lines:
        if line.startswith("--- a/"):
            current_file = line[6:].split("\t")[0]
        elif line.startswith("+++ b/"):
            current_file = line[6:].split("\t")[0]
        elif line.startswith("-") and not line.startswith("---"):
            removals.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            additions.append(line[1:])
    
    if not current_file or not (removals or additions):
        return None
    
    file_path = workdir / current_file
    if not file_path.exists():
        return None
    
    try:
        original = file_path.read_text()
        modified = original
        
        # Try to find and replace the removed block with the added block
        if removals:
            old_block = "\n".join(removals)
            new_block = "\n".join(additions) if additions else ""
            
            if old_block in original:
                modified = original.replace(old_block, new_block, 1)
                changes_made = True
            else:
                # Try line-by-line fuzzy matching
                for old_line in removals:
                    old_stripped = old_line.strip()
                    if old_stripped and old_stripped in original:
                        # Find the index of matching additions
                        idx = removals.index(old_line)
                        new_line = additions[idx] if idx < len(additions) else ""
                        # Replace preserving indentation
                        for orig_line in original.split("\n"):
                            if orig_line.strip() == old_stripped:
                                indent = len(orig_line) - len(orig_line.lstrip())
                                new_with_indent = " " * indent + new_line.strip()
                                modified = modified.replace(orig_line, new_with_indent, 1)
                                changes_made = True
                                break
        
        if changes_made and modified != original:
            file_path.write_text(modified)
            state.budget.patch_attempts += 1
            return ExecResult(
                status="ok",
                summary=f"Applied structured edit to {current_file}",
                artifacts=[current_file],
            )
    except Exception as e:
        logger.warning(f"Structured edit failed: {e}")
    
    return None

def _repair_patch(diff: str) -> str:
    """Try to repair common patch format issues."""
    lines = diff.split("\n")
    repaired = []
    
    in_header = True
    for i, line in enumerate(lines):
        # Fix missing diff header
        if in_header and line.startswith("---") and i == 0:
            # Check if there's a missing diff --git line
            if i + 1 < len(lines) and lines[i + 1].startswith("+++"):
                # Extract paths, stripping existing a/ or b/ prefixes
                src = line[4:].strip().split("\t")[0]
                dst = lines[i + 1][4:].strip().split("\t")[0]
                
                # Strip existing a/ or b/ prefixes to avoid doubling
                if src.startswith("a/"):
                    src = src[2:]
                if dst.startswith("b/"):
                    dst = dst[2:]
                
                if not any(ln.startswith("diff --git") for ln in lines[:i]):
                    repaired.append(f"diff --git a/{src} b/{dst}")
        
        if line.startswith("@@"):
            in_header = False
        
        repaired.append(line)
    
    # Ensure proper line endings
    result = "\n".join(repaired)
    if not result.endswith("\n"):
        result += "\n"
    
    return result


def _try_direct_edit(state: AgentState, proposal: Proposal, diff: str) -> ExecResult | None:
    """Try to apply changes directly to files when patch format is invalid."""
    workdir = Path(state.repo.workdir)
    
    # Look for file and content in the diff
    files = _extract_files_from_diff(diff)
    if not files:
        return None
    
    # Try to extract the actual changes from the diff
    # Look for +/- lines that aren't headers
    additions = []
    deletions = []
    
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            additions.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            deletions.append(line[1:])
    
    if not additions and not deletions:
        return None
    
    # Combine deletions and additions into blocks
    old_block = "\n".join(deletions)
    new_block = "\n".join(additions)
    
    # Try to find and modify the file
    for fname in files:
        fpath = workdir / fname
        if not fpath.exists():
            continue
        
        try:
            content = fpath.read_text()
            modified = content
            
            # Strategy 1: Replace exact block
            if old_block and old_block in modified:
                modified = modified.replace(old_block, new_block, 1)
            
            # Strategy 2: Try line-by-line replacements
            elif deletions:
                for i, old_line in enumerate(deletions):
                    if old_line and old_line in modified:
                        new_line = additions[i] if i < len(additions) else ""
                        modified = modified.replace(old_line, new_line, 1)
            
            # Strategy 3: Just insert additions after a context line
            elif additions and not deletions:
                # Look for context in the diff (lines without +/-)
                context_lines = [l for l in diff.split("\n") 
                                if l and not l.startswith(("+", "-", "@", "diff", "---", "+++"))]
                for ctx in context_lines[:3]:
                    if ctx.strip() and ctx.strip() in content:
                        # Insert after context
                        idx = content.find(ctx.strip()) + len(ctx.strip())
                        modified = content[:idx] + "\n" + new_block + content[idx:]
                        break
            
            if modified != content:
                fpath.write_text(modified)
                state.budget.patch_attempts += 1
                return ExecResult(
                    status="ok",
                    summary=f"Applied direct edit to {fname}",
                    artifacts=[fname],
                )
        except Exception:
            continue
    
    return None


def _exec_run_tests(state: AgentState, proposal: Proposal) -> ExecResult:
    """Execute a test run proposal."""
    workdir = Path(state.repo.workdir)
    command = proposal.inputs.get("command", "pytest")
    
    cmd_parts = command.split() if isinstance(command, str) else command
    
    try:
        rc, out, err = _run_argv(workdir, list(cmd_parts), timeout_sec=300)
        passed = (rc == 0)
        
        if not passed:
            failures = _extract_test_failures((out or "") + (err or ""))
            
            from agent.types import TestFailure
            state.last_failures = [
                TestFailure(nodeid=f["nodeid"], message=f["message"])
                for f in failures[:5]
            ]
        else:
            state.last_failures = []
        
        return ExecResult(
            status="ok" if passed else "fail",
            summary=f"Tests {'passed' if passed else 'failed'}",
            artifacts=[],
            metrics={
                "test_result": {
                    "passed": passed,
                    "returncode": rc,
                    "stdout_tail": (out or "")[-1000:],
                }
            },
        )
    except subprocess.TimeoutExpired:
        return ExecResult(status="fail", summary="Test run timed out (5 min)")
    except Exception as e:
        return ExecResult(status="fail", summary=f"Test run failed: {e}")


def _extract_test_failures(output: str) -> list[dict]:
    """Extract detailed failure information from pytest output.
    
    Returns:
        List of dictionaries with 'nodeid' and 'message' keys
    """
    failures = []
    lines = output.split("\n")
    
    current_test = None
    current_message = []
    in_failure_block = False
    
    for i, line in enumerate(lines):
        # Detect FAILED lines
        if "FAILED" in line and "::" in line:
            # Save previous failure
            if current_test:
                failures.append({
                    "nodeid": current_test,
                    "message": "\n".join(current_message[-10:])  # Last 10 lines
                })
            
            # Extract test name
            parts = line.split()
            for part in parts:
                if "::" in part:
                    current_test = part.strip()
                    break
            current_message = []
            in_failure_block = True
        
        # Capture assertion errors
        elif "AssertionError" in line or "assert " in line.lower():
            current_message.append(line.strip())
        
        # Capture error lines
        elif line.strip().startswith("E "):
            current_message.append(line.strip()[2:])  # Remove "E " prefix
        
        # Capture the actual error message
        elif "Error:" in line or "Exception:" in line:
            current_message.append(line.strip())
    
    # Don't forget the last failure
    if current_test:
        failures.append({
            "nodeid": current_test,
            "message": "\n".join(current_message[-10:])
        })
    
    # Fallback: simple FAILED line extraction
    if not failures:
        for line in lines:
            if "FAILED" in line:
                failures.append({
                    "nodeid": line.strip(),
                    "message": line.strip()
                })
    
    return failures


def _exec_finalize(state: AgentState, proposal: Proposal) -> ExecResult:
    """Execute a finalize proposal."""
    summary = proposal.inputs.get("summary", "")
    status = proposal.inputs.get("status", "complete")
    
    state.notes["solved"] = status == "complete"
    state.notes["final_summary"] = summary
    
    return ExecResult(
        status="ok",
        summary=f"Task finalized: {status}",
        metrics={"final_summary": summary, "completion_status": status},
    )
