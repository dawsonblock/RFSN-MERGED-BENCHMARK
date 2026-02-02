from __future__ import annotations
from rfsn_controller.no_llm_guard import assert_llm_allowed
"""Gemini API client with structured output enforcement for RFSN controller."""

import os

MODEL = "gemini-2.0-flash"

# Lazy import: only import google.genai when actually calling the model
# This allows the controller to be imported even if google-genai is not installed
_genai = None
_types = None


def _ensure_genai_imported():
    """Lazily import google.genai modules."""
    global _genai, _types
    if _genai is None:
        try:
            from google import genai
            from google.genai import types as genai_types

            _genai = genai
            _types = genai_types
        except ImportError as e:
            # Raise a clear error instructing users to install optional LLM dependencies.
            raise RuntimeError(
                "Google GenAI SDK not available. To enable this provider, install the optional LLM dependencies "
                "via requirements-llm.txt, e.g. pip install -r requirements-llm.txt"
            ) from e
    return _genai, _types


SYSTEM = """
You are RFSN-CODE, a controller-governed CODING AGENT operating inside a locked-down sandbox.

You do not have direct filesystem or execution authority.
You can only act by emitting valid JSON in one of these modes:
- "tool_request"
- "patch"
- "feature_summary" (only when feature mode is active)

Anything outside JSON is invalid.

========================
NON-NEGOTIABLE REALITY
========================
1) NO SHELL. Commands run with shell=False.
   Never use: cd, &&, ||, |, >, <, $(...), backticks, newlines in commands, inline env vars like FOO=1 cmd.
   Commands must be single, direct invocations that work from repo root.
   If you need multiple steps, request multiple tool calls.

2) COMMAND ALLOWLIST IS LAW.
   A command being "common" does not mean it can run.
   If a command is blocked/denied, do not retry or workaround. Adapt immediately.

3) TOOL QUOTAS EXIST.
   Each tool_request must be high-value.
   Avoid repeated greps/reads. Do not ask for the same info twice.

4) PATCH HYGIENE EXISTS AND IS MODE-DEPENDENT.
   Repair mode expects very small diffs.
   Feature mode permits larger diffs, multiple files, tests, and docs when required.
   Forbidden dirs and secret patterns are never allowed in any mode.

========================
YOUR OBJECTIVE
========================
Implement correct, minimal, verifiable changes in a repository.

Default Definition of Done:
- Behavior matches the task / acceptance criteria
- Verification exists (tests preferred; smoke/contract check acceptable if tests are weak)
- Existing verification passes
- Minimal diffs with no unrelated refactors or formatting churn
- Predictable error handling

If repo docs define stricter rules, follow them.

========================
OUTPUT CONTRACT (STRICT)
========================

TOOL_REQUEST
{
  "mode": "tool_request",
  "requests": [
    {"tool": "sandbox.read_file", "args": {"path": "README.md"}}
  ],
  "why": "What evidence you need and how it drives the next patch."
}

PATCH
{
  "mode": "patch",
  "diff": "unified diff here",
  "why": "What changed, why it's necessary, and what verification will prove it."
}

FEATURE_SUMMARY (feature mode only)
{
  "mode": "feature_summary",
  "summary": "What was built, where it lives, how to use it, what was verified.",
  "completion_status": "complete|partial|blocked|in_progress"
}

Rules:
- Output ONLY the JSON object.
- Always include "why" in tool_request and patch.
- Never claim success without verification evidence.

========================
MANDATORY WORKFLOW
========================
You must follow this sequence unless the controller state already contains the evidence.

1) Establish ground truth
   - If failures exist: identify failing tests and the minimal repro command.
   - If feature work: identify the relevant entry points, modules, and patterns.

2) Inspect narrowly
   - Prefer one grep to locate the owning code.
   - Read only the smallest necessary files.
   - If unclear, read existing tests first.

3) Plan (briefly, in "why")
   - State what is broken/missing (one sentence).
   - State the smallest change you will make.
   - State the verification you will run.

4) Implement via patch
   - Keep diffs targeted.
   - Match repo conventions.
   - Add/update tests when behavior changes.

5) Verify
   - Run focused verification first when possible.
   - Then run full verification required by repo norms.

6) Stop
   - Stop changing code when done criteria are satisfied.
   - In feature mode, emit feature_summary only after verification.

========================
ALLOWLIST-FIRST BEHAVIOR (CRITICAL FIX)
========================
You must treat allowlist uncertainty as a first-class constraint.

- If the project appears Node/Rust/Go/Java/.NET:
  - First read README/CI scripts to learn exact commands.
  - Then issue a tool_request for the exact single-step commands required.
  - If a required command is blocked, declare BLOCKED with the exact command name and why it is necessary.

Never assume multi-language support based on detection alone.
Only trust what the sandbox successfully executes.

========================
SHELL-LESS COMMAND RULES (CRITICAL FIX)
========================
You must never propose compound commands.

Bad (never do):
- "npm install && npm test"
- "cd pkg && pytest"
- "FOO=1 pytest"
- "pytest | tee out.txt"

Good:
- separate tool calls:
  - "npm install"
  - "npm test"
- or explicit paths:
  - "python -m pytest tests/test_x.py"

If you need environment variables, modify config files or pass supported flags instead of inline env assignment.

========================
FEATURE-MODE VERIFICATION RULES (CRITICAL FIX)
========================
Feature completion is not declarative.

completion_status="complete" is allowed ONLY when you have at least one hard verification result:
- tests passed, OR
- a smoke command ran successfully with expected output, OR
- a contract check executed successfully (import + callable + invariant)

If verification has not been executed, you must use:
- "in_progress" or "partial"
If blocked by tooling/allowlist/deps:
- "blocked" with the concrete blocker

Never mark complete based on reasoning alone.

========================
HYGIENE PROFILE BEHAVIOR (CRITICAL FIX)
========================
You must adapt your patch strategy to the task mode.

REPAIR MODE:
- Keep changes extremely small.
- Touch minimal files.
- Avoid adding new modules unless necessary.
- NEVER edit test files. Fix the SOURCE/IMPLEMENTATION code, not the tests.
- The bug is in the implementation, not the test. Patch the source file.

FEATURE MODE:
- Multiple files are acceptable when functionally required.
- New modules + tests + docs are expected.
- Still avoid unrelated refactors, renames, and formatting churn.
- If hygiene gates block legitimate work, you must reduce scope or declare the constraint explicitly.

Always forbidden in all modes:
- touching forbidden directories (vendor/, third_party/, node_modules/, .venv/, dist/, build/, target/, etc.)
- touching secrets/keys/env files or credential patterns
- lockfile churn unless repo evidence shows it's required AND the controller permits it

========================
STALL / RETRY POLICY
========================
If a command is blocked, a patch is hygiene-rejected, or verification fails:
- Do not repeat the same action.
- Switch to new evidence gathering or a different minimal strategy.
- If no valid path exists, declare BLOCKED with evidence.

You are a bounded coding agent. Act through evidence, minimal diffs, and verification.
""".strip()


def _build_schemas():
    """Build the Gemini API schemas using lazy-imported types."""
    _, types = _ensure_genai_imported()

    # Schema: mode + either requests or diff
    request_item = types.Schema(
        type=types.Type.OBJECT,
        required=["tool", "args"],
        properties={
            "tool": types.Schema(type=types.Type.STRING),
            "args": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "path": types.Schema(type=types.Type.STRING),
                    "cmd": types.Schema(type=types.Type.STRING),
                    "query": types.Schema(type=types.Type.STRING),
                    "github_url": types.Schema(type=types.Type.STRING),
                    "diff": types.Schema(type=types.Type.STRING),
                    "ref": types.Schema(type=types.Type.STRING),
                    "max_bytes": types.Schema(type=types.Type.INTEGER),
                    "max_matches": types.Schema(type=types.Type.INTEGER),
                    "max_files": types.Schema(type=types.Type.INTEGER),
                    "timeout_sec": types.Schema(type=types.Type.INTEGER),
                    "packages": types.Schema(type=types.Type.STRING),
                    "requirements_file": types.Schema(type=types.Type.STRING),
                    "venv_path": types.Schema(type=types.Type.STRING),
                    "module_name": types.Schema(type=types.Type.STRING),
                },
            ),
        },
    )

    output_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["mode"],
        properties={
            "mode": types.Schema(
                type=types.Type.STRING, enum=["tool_request", "patch", "feature_summary"]
            ),
            "requests": types.Schema(type=types.Type.ARRAY, items=request_item),
            "why": types.Schema(type=types.Type.STRING),
            "diff": types.Schema(type=types.Type.STRING),
            "summary": types.Schema(type=types.Type.STRING),
            "completion_status": types.Schema(type=types.Type.STRING),
        },
    )

    return output_schema


_client = None  # cached client instance


def client():
    """Return a singleton Google GenAI client, reading API key from env.

    Raises:
        RuntimeError: if the GEMINI_API_KEY environment variable is not set or google-genai is not installed.
    """
    global _client
    if _client is None:
        genai, _ = _ensure_genai_imported()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        _client = genai.Client(api_key=key)
    return _client


def call_model(model_input: str, temperature: float = 0.0) -> dict:
    """Call the Gemini model with structured JSON output enforcement.

    Args:
        model_input: The text prompt to send to the model.
        temperature: Sampling temperature for creative variance.

    Returns:
        A dictionary parsed from the JSON response. It always contains
        at least a "mode" key and may include "requests", "why", or "diff".

    Raises:
        RuntimeError: if google-genai SDK is not available.
    """
    import time as _time
    start_time = _time.time()
    
    genai, types = _ensure_genai_imported()
    output_schema = _build_schemas()

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=output_schema,
    )
    
    error_msg = None
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        resp = client().models.generate_content(
            model=MODEL,
            contents=model_input,
            config=cfg,
        )
        
        # Extract token usage from response
        if hasattr(resp, "usage_metadata"):
            prompt_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0)
        
        # Track LLM call and tokens in budget
        try:
            from ..budget import record_llm_call_global
            total_tokens = prompt_tokens + completion_tokens
            record_llm_call_global(tokens=total_tokens)
        except ImportError:
            pass  # Budget module not available
        
        data = getattr(resp, "parsed", None)
        success = True
        
    except Exception as e:
        error_msg = str(e)
        success = False
        data = None
    
    # Log LLM call event
    latency_ms = (_time.time() - start_time) * 1000
    try:
        from ..events import log_llm_call_global
        log_llm_call_global(
            model=MODEL,
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error_msg,
        )
    except ImportError:
        pass  # Events module not available
    
    if error_msg:
        raise RuntimeError(f"Gemini API call failed: {error_msg}")
    
    if isinstance(data, dict) and "mode" in data:
        return data
    # fallback: treat as patch with empty diff if parsing failed
    return {"mode": "patch", "diff": ""}


async def call_model_async(model_input: str, temperature: float = 0.0) -> dict:
    """Async version of call_model."""
    import asyncio
    import time
    
    genai, types = _ensure_genai_imported()
    output_schema = _build_schemas()

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=output_schema,
    )
    
    max_retries = 3
    base_delay = 1.0
    last_exception = None
    
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            # correct async call for google-genai v1 sdk
            resp = await client().aio.models.generate_content(
                model=MODEL,
                contents=model_input,
                config=cfg,
            )
            
            data = getattr(resp, "parsed", None)
            
            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000
            # Extract token usage
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(resp, "usage_metadata"):
                prompt_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0)
            
            try:
                from ..telemetry import track_llm_call
                track_llm_call(
                    model=MODEL,
                    status="success",
                    latency_sec=latency_sec,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except ImportError:
                pass
            
            # Track LLM call and tokens in budget
            try:
                from ..budget import record_llm_call_global
                total_tokens = prompt_tokens + completion_tokens
                record_llm_call_global(tokens=total_tokens)
            except ImportError:
                pass  # Budget module not available
            
            # Log LLM call event
            try:
                from ..events import log_llm_call_global
                log_llm_call_global(
                    model=MODEL,
                    tokens_prompt=prompt_tokens,
                    tokens_completion=completion_tokens,
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                )
            except ImportError:
                pass  # Events module not available

            if isinstance(data, dict) and "mode" in data:
                return data
            return {"mode": "patch", "diff": ""}
            
        except Exception as e:
            last_exception = e
            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000
            try:
                from ..telemetry import track_llm_call
                track_llm_call(
                    model=MODEL,
                    status="error",
                    latency_sec=latency_sec,
                )
            except ImportError:
                pass
            
            # Log failed LLM call event
            try:
                from ..events import log_llm_call_global
                log_llm_call_global(
                    model=MODEL,
                    tokens_prompt=0,
                    tokens_completion=0,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )
            except ImportError:
                pass  # Events module not available
            
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                raise last_exception from last_exception
                
    raise last_exception if last_exception else RuntimeError("Unknown error")


async def call_model_streaming(model_input: str, temperature: float = 0.0):
    """Call the Gemini model with streaming response.
    
    Yields:
        Chunks of the response content as they arrive.
    """
    genai, types = _ensure_genai_imported()
    output_schema = _build_schemas()

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=output_schema,
    )
    
    try:
        # correct async call for google-genai v1 sdk with streaming
        # Note: aio.models.generate_content(stream=True) returns an async iterator
        async for chunk in await client().aio.models.generate_content(
            model=MODEL,
            contents=model_input,
            config=cfg,
            stream=True
        ):
            # Gemini chunks might contain parts or text
            if chunk.text:
                yield chunk.text
            
    except Exception as e:
        raise e

