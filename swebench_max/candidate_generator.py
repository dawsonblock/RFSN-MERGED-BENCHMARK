"""
Candidate Generator with:
- 3-planner ensemble (primary, alt, skeptic)
- Patch dedup by normalized diff hash
- Skeptic rewrite loop
- Retrieval from past failure fingerprints
"""
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from swebench_max.candidate import Candidate


@dataclass
class PlannerConfig:
    name: str
    weight: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class GeneratorState:
    """Mutable state for the generator across rounds."""
    seen_hashes: Set[str] = field(default_factory=set)
    failed_patches: List[str] = field(default_factory=list)
    round_idx: int = 0


def normalize_patch(patch: str) -> str:
    """
    Normalize patch for dedup comparison.
    Strips whitespace variations, sorts hunks.
    """
    lines = []
    for line in patch.splitlines():
        line = line.rstrip()
        # Skip diff metadata that varies
        if line.startswith("index ") or line.startswith("@@"):
            continue
        lines.append(line)
    return "\n".join(sorted(lines))


def hash_patch(patch: str) -> str:
    """Hash normalized patch for dedup."""
    norm = normalize_patch(patch)
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def _build_prompt(
    issue: Dict[str, Any],
    planner: PlannerConfig,
    failed_patches: List[str],
    file_context: str,
) -> str:
    """Build the prompt for the LLM based on planner type."""
    
    problem = issue.get("problem_statement", "Fix the issue.")
    repo = issue.get("repo", "unknown")
    
    base_prompt = f"""You are an expert software engineer fixing a bug in {repo}.

## Problem Statement
{problem}

## Relevant File Context
{file_context}

## Instructions
1. Analyze the problem carefully
2. Generate a minimal, focused patch in unified diff format
3. Only modify what's necessary to fix the issue
4. Ensure the patch applies cleanly

Output ONLY the unified diff patch, nothing else.
"""
    
    if planner.name == "skeptic" and failed_patches:
        # Skeptic sees what failed and tries different approach
        failed_summary = "\n---\n".join(failed_patches[:3])
        base_prompt += f"""

## Previous Failed Approaches (DO NOT REPEAT)
The following patches were tried and failed. Take a DIFFERENT approach:
{failed_summary}

Think about why these failed and try an alternative solution.
"""
    
    elif planner.name == "alt":
        base_prompt += """

## Alternative Approach
Consider an unconventional or alternative solution to this problem.
Think about edge cases and less obvious fixes.
"""
    
    return base_prompt


def _call_llm(
    prompt: str,
    planner: PlannerConfig,
    cfg: Dict[str, Any],
) -> Optional[str]:
    """
    Call the LLM to generate a patch.
    Uses the configured LLM client.
    """
    try:
        # Try to use the configured LLM
        llm_cfg = cfg.get("llm", {})
        provider = llm_cfg.get("provider", "deepseek")
        
        if provider == "deepseek":
            import httpx
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                return None
            
            # Use deepseek-reasoner for better code reasoning
            model = llm_cfg.get("model", "deepseek-reasoner")
            
            resp = httpx.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": planner.temperature if model != "deepseek-reasoner" else 0,  # reasoner requires temp=0
                    "max_tokens": planner.max_tokens,
                },
                timeout=300.0,  # Reasoner needs more time
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            # reasoner returns reasoning_content + content; we want the final content
            return message.get("content", "") or message.get("reasoning_content", "")
        
        elif provider == "openai":
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=llm_cfg.get("model", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                temperature=planner.temperature,
                max_tokens=planner.max_tokens,
            )
            return resp.choices[0].message.content
        
        else:
            # Fallback: no LLM configured
            return None
            
    except Exception as e:
        print(f"LLM call failed for {planner.name}: {e}")
        return None


def _extract_patch(response: str) -> Optional[str]:
    """Extract unified diff from LLM response."""
    if not response:
        return None
    
    lines = response.splitlines()
    patch_lines = []
    in_patch = False
    
    for line in lines:
        # Start of diff
        if line.startswith("---") or line.startswith("diff --git"):
            in_patch = True
        
        if in_patch:
            patch_lines.append(line)
        
        # Also capture if wrapped in code block
        if line.strip() == "```diff":
            in_patch = True
            continue
        if line.strip() == "```" and in_patch:
            break
    
    if not patch_lines:
        # Try to find any diff-like content
        for i, line in enumerate(lines):
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                # Found start of diff
                for j in range(max(0, i-1), len(lines)):
                    if lines[j].startswith("diff") or lines[j].startswith("---"):
                        patch_lines = lines[j:]
                        break
                break
    
    if not patch_lines:
        return None
    
    return "\n".join(patch_lines)


def _get_file_context(issue: Dict[str, Any], repo_root: str) -> str:
    """
    Retrieve relevant file context for the issue.
    Uses hints from the issue or searches for likely files.
    """
    context_parts = []
    
    # Check if issue has hints about files
    hints = issue.get("hints", {})
    files = hints.get("files", [])
    
    if not files:
        # Try to extract from problem statement
        problem = issue.get("problem_statement", "")
        # Look for file paths mentioned
        import re
        file_refs = re.findall(r'[\w/]+\.py', problem)
        files = list(set(file_refs))[:5]
    
    for fpath in files[:5]:
        full_path = os.path.join(repo_root, fpath)
        if os.path.isfile(full_path):
            try:
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Truncate if too long
                if len(content) > 10000:
                    content = content[:10000] + "\n... (truncated)"
                context_parts.append(f"### {fpath}\n```python\n{content}\n```")
            except Exception:
                pass
    
    return "\n\n".join(context_parts) if context_parts else "(no file context available)"


def _load_failure_fingerprints(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Load past failure fingerprints from cache.
    Returns mapping of error signature -> patches that fixed similar issues.
    """
    cache_dir = cfg.get("cache", {}).get("dir", ".rfsn_cache")
    fp_file = os.path.join(cache_dir, "failure_fingerprints.json")
    
    if os.path.isfile(fp_file):
        try:
            with open(fp_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    
    return {}


def _save_failure_fingerprint(
    cfg: Dict[str, Any],
    error_sig: str,
    patch: str,
    success: bool,
):
    """Save a failure fingerprint for future retrieval."""
    cache_dir = cfg.get("cache", {}).get("dir", ".rfsn_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fp_file = os.path.join(cache_dir, "failure_fingerprints.json")
    
    fingerprints = _load_failure_fingerprints(cfg)
    
    if success:
        # Add successful patch for this error signature
        if error_sig not in fingerprints:
            fingerprints[error_sig] = []
        fingerprints[error_sig].append(patch)
        fingerprints[error_sig] = fingerprints[error_sig][-10:]  # Keep last 10
    
    try:
        with open(fp_file, "w", encoding="utf-8") as f:
            json.dump(fingerprints, f, indent=2)
    except Exception:
        pass


def _extract_error_signature(issue: Dict[str, Any]) -> str:
    """Extract a signature from the error for fingerprint matching."""
    problem = issue.get("problem_statement", "")
    # Simple: hash first 500 chars of problem
    return hashlib.sha256(problem[:500].encode()).hexdigest()[:12]


def generate_candidates(
    issue: Dict[str, Any],
    cfg: Dict[str, Any],
    state: Optional[GeneratorState] = None,
    repo_root: str = ".",
) -> List[Candidate]:
    """
    Generate candidate patches using ensemble of planners.
    
    Features:
    - 3-planner ensemble: primary, alt, skeptic
    - Patch dedup by normalized diff hash
    - Skeptic rewrite loop using failed patches
    - Retrieval from past failure fingerprints
    """
    if state is None:
        state = GeneratorState()
    
    candidates: List[Candidate] = []
    
    # Load planner configs
    planner_cfgs = cfg.get("planners", [
        {"name": "primary", "weight": 1.0},
        {"name": "alt", "weight": 0.7},
        {"name": "skeptic", "weight": 0.5},
    ])
    planners = [PlannerConfig(**p) for p in planner_cfgs]
    
    # Get file context
    file_context = _get_file_context(issue, repo_root)
    
    # Check for relevant failure fingerprints
    error_sig = _extract_error_signature(issue)
    fingerprints = _load_failure_fingerprints(cfg)
    prior_fixes = fingerprints.get(error_sig, [])
    
    # If we have prior successful fixes, add them as candidates first
    for i, prior_patch in enumerate(prior_fixes[:2]):
        patch_hash = hash_patch(prior_patch)
        if patch_hash not in state.seen_hashes:
            state.seen_hashes.add(patch_hash)
            candidates.append(Candidate(
                key=f"fingerprint_{i}_{patch_hash}",
                patch=prior_patch,
                meta={"source": "fingerprint", "round": state.round_idx},
            ))
    
    # Generate from each planner
    for planner in planners:
        # Build prompt
        prompt = _build_prompt(issue, planner, state.failed_patches, file_context)
        
        # Call LLM
        response = _call_llm(prompt, planner, cfg)
        
        if not response:
            continue
        
        # Extract patch
        patch = _extract_patch(response)
        
        if not patch:
            continue
        
        # Dedup by hash
        patch_hash = hash_patch(patch)
        if patch_hash in state.seen_hashes:
            continue
        
        state.seen_hashes.add(patch_hash)
        
        # Create candidate
        candidate = Candidate(
            key=f"{planner.name}_r{state.round_idx}_{patch_hash}",
            patch=patch,
            meta={
                "planner": planner.name,
                "weight": planner.weight,
                "round": state.round_idx,
            },
        )
        candidates.append(candidate)
    
    # Skeptic rewrite loop: if we have failed patches, generate more skeptic variants
    if state.failed_patches and state.round_idx > 0:
        skeptic_planner = PlannerConfig(
            name="skeptic_rewrite",
            weight=0.6,
            temperature=0.9,  # Higher temp for more variety
        )
        
        for attempt in range(2):
            prompt = _build_prompt(issue, skeptic_planner, state.failed_patches, file_context)
            prompt += f"\n\nThis is rewrite attempt {attempt + 1}. Be creative."
            
            response = _call_llm(prompt, skeptic_planner, cfg)
            patch = _extract_patch(response) if response else None
            
            if patch:
                patch_hash = hash_patch(patch)
                if patch_hash not in state.seen_hashes:
                    state.seen_hashes.add(patch_hash)
                    candidates.append(Candidate(
                        key=f"skeptic_rewrite_{attempt}_{patch_hash}",
                        patch=patch,
                        meta={
                            "planner": "skeptic_rewrite",
                            "attempt": attempt,
                            "round": state.round_idx,
                        },
                    ))
    
    # Shuffle to avoid bias from planner order
    random.shuffle(candidates)
    
    state.round_idx += 1
    
    return candidates


def record_candidate_result(
    candidate: Candidate,
    success: bool,
    state: GeneratorState,
    issue: Dict[str, Any],
    cfg: Dict[str, Any],
):
    """
    Record the result of a candidate evaluation.
    Updates failure fingerprints and failed_patches list.
    """
    if not success:
        state.failed_patches.append(candidate.patch)
        # Keep only recent failures
        state.failed_patches = state.failed_patches[-10:]
    else:
        # Save successful patch for future fingerprint retrieval
        error_sig = _extract_error_signature(issue)
        _save_failure_fingerprint(cfg, error_sig, candidate.patch, success=True)
