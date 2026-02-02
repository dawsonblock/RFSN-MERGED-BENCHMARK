from __future__ import annotations
from rfsn_controller.no_llm_guard import assert_llm_allowed
"""DeepSeek API client with structured output enforcement for RFSN controller."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

# Lazy import: only import openai when actually calling the model
# This allows the controller to be imported even if openai is not installed
_openai = None

# =============================================================================
# LLM RESPONSE CACHING
# =============================================================================
# Set RFSN_LLM_CACHE=1 to enable caching, or RFSN_LLM_CACHE=/path/to/cache.db
_cache_db: sqlite3.Connection | None = None
_cache_enabled: bool = os.environ.get("RFSN_LLM_CACHE", "0") not in ("0", "")


def _get_cache_db() -> sqlite3.Connection | None:
    """Get or create the cache database connection."""
    global _cache_db
    if not _cache_enabled:
        return None
    if _cache_db is None:
        cache_path = os.environ.get("RFSN_LLM_CACHE", "")
        if cache_path in ("1", "true", "yes"):
            cache_path = str(Path.home() / ".cache" / "rfsn" / "llm_cache.db")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        _cache_db = sqlite3.connect(cache_path, check_same_thread=False)
        _cache_db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT,
                created_at REAL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        _cache_db.commit()
    return _cache_db


def _cache_key(prompt: str, temperature: float) -> str:
    """Generate a cache key from prompt and temperature."""
    content = f"{prompt}|{temperature}"
    return hashlib.sha256(content.encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    """Get a cached response."""
    db = _get_cache_db()
    if db is None:
        return None
    try:
        cursor = db.execute("SELECT response FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            db.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
            db.commit()
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _cache_set(key: str, response: dict) -> None:
    """Store a response in cache."""
    db = _get_cache_db()
    if db is None:
        return
    try:
        import time
        db.execute(
            "INSERT OR REPLACE INTO cache (key, response, created_at, hit_count) VALUES (?, ?, ?, 0)",
            (key, json.dumps(response), time.time())
        )
        db.commit()
    except Exception:
        pass


def _ensure_openai_imported():
    """Lazily import openai module."""
    global _openai
    if _openai is None:
        try:
            from openai import AsyncOpenAI, OpenAI

            _openai = (OpenAI, AsyncOpenAI)
        except ImportError as e:
            # Raise a clear error instructing users to install optional LLM dependencies.
            raise RuntimeError(
                "OpenAI SDK not available. To enable this provider, install the optional LLM dependencies "
                "via requirements-llm.txt, e.g. pip install -r requirements-llm.txt"
            ) from e
    return _openai


# =============================================================================
# MULTI-MODEL FALLBACK CONFIGURATION
# =============================================================================
# Primary model and fallbacks. Set RFSN_FALLBACK_MODELS to customize.
# Format: comma-separated list of "model:base_url:api_key_env"
# Example: RFSN_FALLBACK_MODELS="gpt-4o:https://api.openai.com/v1:OPENAI_API_KEY"

MODEL = "deepseek-chat"

# Fallback models (tried in order if primary fails)
FALLBACK_MODELS = [
    {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    # Add more fallbacks as needed - these are only used if env vars are set
    {
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    {
        "model": "claude-3-5-sonnet-20241022",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
]


def _get_available_models() -> list[dict]:
    """Get list of models with available API keys."""
    available = []
    for model_config in FALLBACK_MODELS:
        api_key = os.environ.get(model_config["api_key_env"])
        if api_key:
            available.append({**model_config, "api_key": api_key})
    return available

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
- Do not edit tests unless explicitly required.

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

_client = None  # cached client instance
_async_client = None # cached async client instance

class MockClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                assert_llm_allowed()

                class Message:
                    content = '{"mode": "tool_request", "requests": [], "why": "Mocked response because API key is missing."}'
                class Choice:
                    message = Message()
                class Response:
                    choices = [Choice()]
                    usage = None
                return Response()

class AsyncMockClient:
    class chat:
        class completions:
            @staticmethod
            async def create(*args, **kwargs):
                assert_llm_allowed()

                class Message:
                    content = '{"mode": "tool_request", "requests": [], "why": "Mocked async response"}'
                class Choice:
                    message = Message()
                class Response:
                    choices = [Choice()]
                    usage = None
                return Response()

def client():
    """Return a singleton DeepSeek client, reading API key from env.

    Raises:
        RuntimeError: if the DEEPSEEK_API_KEY environment variable is not set or openai SDK is not installed.
    """
    global _client
    if _client is None:
        OpenAI, _ = _ensure_openai_imported()

        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            print("Warning: DEEPSEEK_API_KEY not found. Using Mock Client.")
            _client = MockClient()
        else:
            _client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _client

def async_client():
    """Return a singleton Async DeepSeek client."""
    global _async_client
    if _async_client is None:
        _, AsyncOpenAI = _ensure_openai_imported()
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            _async_client = AsyncMockClient()
        else:
            _async_client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _async_client


def call_model(model_input: str, temperature: float = 0.0) -> dict:
    """Call the DeepSeek model with structured JSON output enforcement.

    Args:
        model_input: The text prompt to send to the model.
        temperature: Sampling temperature for creative variance.

    Returns:
        A dictionary parsed from the JSON response. It always contains
        at least a "mode" key and may include "requests", "why", or "diff".

    Raises:
        RuntimeError: if openai SDK is not available.
    """
    import time
    
    # Check cache first (only for temperature=0 to avoid stale creative responses)
    if temperature == 0.0:
        cache_k = _cache_key(model_input, temperature)
        cached = _cache_get(cache_k)
        if cached is not None:
            return cached
    
    _ensure_openai_imported()  # Ensure SDK is available before making the call

    # Retry with exponential backoff for transient failures
    max_retries = 3
    base_delay = 1.0
    last_exception = None
    cache_k = _cache_key(model_input, temperature) if temperature == 0.0 else None
    
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            resp = client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": model_input},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            content = resp.choices[0].message.content
            result = json.loads(content)
            
            # Track successful call with telemetry
            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000
            prompt_tokens = getattr(resp.usage, 'prompt_tokens', 0) if hasattr(resp, 'usage') else 0
            completion_tokens = getattr(resp.usage, 'completion_tokens', 0) if hasattr(resp, 'usage') else 0
            
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
                pass  # Telemetry not available
            
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
            
            # Cache successful response (only for temperature=0)
            if cache_k is not None:
                _cache_set(cache_k, result)
            
            return result
            
        except Exception as e:
            last_exception = e
            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000
            
            # Track failed call
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
                time.sleep(delay)
            else:
                raise last_exception from last_exception
    
    # Should never reach here
    raise last_exception if last_exception else RuntimeError("Unknown error")


def call_model_with_fallback(model_input: str, temperature: float = 0.0) -> dict:
    """Call LLM with fallback to alternative models if primary fails.
    
    Tries each available model in order until one succeeds.
    
    Args:
        model_input: The text prompt to send to the model.
        temperature: Sampling temperature for creative variance.
        
    Returns:
        A dictionary parsed from the JSON response.
        
    Raises:
        RuntimeError: If all models fail.
    """
    import time
    
    _ensure_openai_imported()
    OpenAI, _ = _openai
    
    available_models = _get_available_models()
    if not available_models:
        raise RuntimeError("No LLM API keys configured. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")
    
    errors = []
    
    for model_config in available_models:
        model_name = model_config["model"]
        base_url = model_config["base_url"]
        api_key = model_config["api_key"]
        
        # Skip Anthropic for now (different API format)
        if "anthropic" in base_url.lower():
            continue
        
        try:
            temp_client = OpenAI(api_key=api_key, base_url=base_url)
            
            start_time = time.time()
            resp = temp_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": model_input},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            
            content = resp.choices[0].message.content
            result = json.loads(content)
            
            latency_sec = time.time() - start_time
            
            # Log success with fallback model
            try:
                from ..telemetry import track_llm_call
                track_llm_call(
                    model=model_name,
                    status="success",
                    latency_sec=latency_sec,
                )
            except ImportError:
                pass
            
            return result
            
        except Exception as e:
            errors.append(f"{model_name}: {e}")
            continue
    
    # All models failed
    raise RuntimeError(f"All LLM models failed: {'; '.join(errors)}")

async def call_model_async(model_input: str, temperature: float = 0.0) -> dict:
    """Async version of call_model."""
    import asyncio
    import time
    
    _ensure_openai_imported()

    max_retries = 3
    base_delay = 1.0
    last_exception = None
    
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            resp = await async_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": model_input},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            content = resp.choices[0].message.content
            result = json.loads(content)
            
            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000
            prompt_tokens = getattr(resp.usage, 'prompt_tokens', 0) if hasattr(resp, 'usage') else 0
            completion_tokens = getattr(resp.usage, 'completion_tokens', 0) if hasattr(resp, 'usage') else 0
            
            try:
                from .telemetry import track_llm_call
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
                from .budget import record_llm_call_global
                total_tokens = prompt_tokens + completion_tokens
                record_llm_call_global(tokens=total_tokens)
            except ImportError:
                pass  # Budget module not available
            
            # Log LLM call event
            try:
                from .events import log_llm_call_global
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
            
            return result
            
        except Exception as e:
            last_exception = e
            latency_sec = time.time() - start_time
            latency_ms = latency_sec * 1000
            try:
                from .telemetry import track_llm_call
                track_llm_call(
                    model=MODEL,
                    status="error",
                    latency_sec=latency_sec,
                )
            except ImportError:
                pass
            
            # Log failed LLM call event
            try:
                from .events import log_llm_call_global
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
    """Call the DeepSeek model with streaming response.
    
    Yields:
        Chunks of the response content as they arrive.
    """
    _ensure_openai_imported()

    try:
        stream = await async_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": model_input},
            ],
            temperature=temperature,
            stream=True,
            response_format={"type": "json_object"},
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
                
    except Exception as e:
        raise e



