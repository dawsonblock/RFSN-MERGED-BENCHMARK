"""LLM Patch Generator with Learning.

Factory for patch-candidate generator with:
1. Thompson Sampling for prompt variant selection
2. Cross-attempt feedback (learns from previous failures)
3. Bandit persistence across runs

The replay avoids any LLM imports/calls for deterministic re-execution.
"""
import os
import logging
from typing import Any, Optional

from rfsn_controller.replay_trace import (
    TraceWriter, TraceReader,
    make_recording_llm_wrapper, make_replay_llm_wrapper
)

logger = logging.getLogger(__name__)

_RFSN_RECORD_TRACE = os.environ.get("RFSN_RECORD_TRACE")
_RFSN_REPLAY_TRACE = os.environ.get("RFSN_REPLAY_TRACE")

_trace_writer = TraceWriter(_RFSN_RECORD_TRACE, run_meta={"mode": "record"}) if _RFSN_RECORD_TRACE else None
_trace_reader = TraceReader(_RFSN_REPLAY_TRACE, verify_chain=True) if _RFSN_REPLAY_TRACE else None


def get_llm_patch_fn(model_name: str = "deepseek"):
    """Factory for patch-candidate generator with learning.
    
    Args:
        model_name: LLM model to use (deepseek/gemini)
        
    Returns:
        Callable that generates patch candidates with learning
    """
    if _trace_reader is not None:
        return make_replay_llm_wrapper(_trace_reader)

    # Lazy import so replay/verification can import this module with LLM disabled.
    from rfsn_upstream.prompt_variants import get_variant, format_prompt, list_variants
    from rfsn_upstream.llm_prompting import extract_diff_from_response, call_llm, LLMConfig, LLMProvider
    from rfsn_upstream.bandit import create_bandit
    
    # Initialize Thompson Sampling bandit for variant selection
    variant_names = list_variants()
    bandit = create_bandit(
        db_path=".rfsn_state/variant_bandit.db",
        arms=variant_names,
    )
    
    # Track attempts across calls (for feedback loop)
    # This is reset per-task by episode_runner
    attempt_history: list[dict[str, Any]] = []

    def _generate_patches(plan: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Default patch generator using upstream prompt variants with learning."""
        nonlocal attempt_history
        
        # --- Prompt variant selection ---
        # Priority:
        #   1) upstream forced variant (if valid)
        #   2) rotate variants on retries for diversity
        #   3) bandit selection
        up = {}
        if isinstance(context, dict):
            up = (
                context.get("upstream_hints")
                or context.get("_upstream_hints")
                or {}
            )

        forced = (
            up.get("prompt_variant")
            or up.get("variant")
            or up.get("prompt_variant_name")
        )
        forced_flag = False

        # Diversity rotation for retries - try different variants each attempt
        diversity_variants = [
            "v_diagnose_then_patch",      # Attempt 1: diagnose first
            "v_multi_hypothesis",          # Attempt 2: multiple hypotheses
            "v_traceback_local",          # Attempt 3: traceback-focused
            "v_api_compat_shim",          # Attempt 4: compatibility shim
            "v_chain_of_thought",         # Attempt 5: reasoning first
            "v_repair_loop",              # Attempt 6: iterative refinement
        ]

        if forced and forced in variant_names:
            variant_name = forced
            forced_flag = True
            logger.info("ENVIRONMENT: Prompt variant forced by upstream: %s", variant_name)
        elif attempt_history:
            # Rotate through diversity variants on retries
            attempt_idx = len(attempt_history) % len(diversity_variants)
            variant_name = diversity_variants[attempt_idx]
            logger.info("ENVIRONMENT: Using diversity variant %s (attempt %d)", 
                        variant_name, len(attempt_history) + 1)
        else:
            variant_name = bandit.select_arm()
            
        # --- Model selection ---
        # Always use DeepSeek for code generation (most reliable for code)
        # Ignore upstream model hints - user explicitly requested deepseek
        model_hint = "reasoner"  # Force deepseek
        provider = LLMProvider.DEEPSEEK
        model_id = "deepseek-reasoner"

        
        variant = get_variant(variant_name)
        
        # Temperature scaling: increase on retries for more diversity
        base_temp = variant.temperature
        attempt_num = len(attempt_history)
        temp_boost = min(0.3, 0.1 * attempt_num)  # Cap at +0.3
        temperature = min(1.0, base_temp + temp_boost)
        
        logger.info("ENVIRONMENT: Selected variant: %s, model: %s, temp: %.2f (attempt %d)", 
                    variant_name, model_hint, temperature, attempt_num + 1)

        
        # 2. Build prompt
        task_info = plan.task if hasattr(plan, "task") else {}
        if not task_info and isinstance(plan, dict):
            task_info = plan.get("task", {})

        problem_statement = task_info.get("problem_statement", "")
        
        # Prepare file content with line numbers
        raw_files = context.get("file_context", {})
        file_content_str = ""
        for path, content in raw_files.items():
            file_content_str += f"\nFile: {path} (line numbers shown on left)\n```python\n{content}\n```\n"

        # Build richer rejection history for multi-turn refinement
        rejection_history_str = "None"
        if attempt_history:
            parts = []
            for h in attempt_history[-3:]:  # Last 3 attempts (more detail)
                parts.append(f"""### Attempt {h['attempt']} (variant: {h['variant']})
**Patch generated:**
```diff
{h.get('patch_text', '<none>')[:800]}
```
**Result:** {h['result']}
**Test output (if failed):**
```
{h.get('test_output', '')[:500]}
```
""")
            rejection_history_str = "\n".join(parts)

        system_prompt, user_prompt = format_prompt(
            variant,
            problem_statement=problem_statement,
            test_output=task_info.get("last_test_output", ""),
            file_content=file_content_str,
            # v_repair_loop specific context
            rejection_history=rejection_history_str,
            test_status="failing",
            patches_applied=len(attempt_history),
        )
        
        # Add repair cards context (similar successful fixes)
        repair_cards = context.get("repair_cards", [])
        if repair_cards:
            from retrieval.repair_cards import format_repair_cards_for_prompt
            repair_cards_block = format_repair_cards_for_prompt(repair_cards)
            user_prompt = user_prompt + "\n" + repair_cards_block
            logger.info("Added %d repair cards to prompt", len(repair_cards))
        
        # 3. Call LLM
        # Use Reasoner model for complex code fixes if selected
        timeout = 180 if model_hint == "reasoner" else 60
        config = LLMConfig(
            provider=provider, 
            model=model_id, 
            temperature=temperature,  # Use computed temperature with retry boost
            timeout=timeout

        )
        response = call_llm(user_prompt, system_prompt=system_prompt, config=config)

        content = response.content
        summary = "Generated patch"
        if response.parsed and isinstance(response.parsed, dict):
            summary = response.parsed.get("summary", summary)
        
        # 4. Extract Diff
        patch_text = extract_diff_from_response(content)
        
        # DEBUG: Log what we're extracting
        logger.info("=" * 60)
        logger.info("LLM RESPONSE (first 2000 chars):")
        logger.info(content[:2000] if content else "<empty>")
        logger.info("=" * 60)
        logger.info("EXTRACTED PATCH (first 1000 chars):")
        logger.info(patch_text[:1000] if patch_text else "<none>")
        logger.info("=" * 60)
        
        # Record this attempt for feedback loop (including full patch for next attempt)
        attempt_history.append({
            "attempt": len(attempt_history) + 1,
            "variant": variant_name,
            "summary": summary,
            "patch_text": patch_text if patch_text else "<no valid patch generated>",
            "result": "pending",  # Will be updated by update_attempt_result
            "test_output": "",  # Will be populated by update_attempt_result
            "forced_variant": forced_flag,  # Track if variant was forced by upstream
            "upstream_prompt_variant": forced if forced_flag else None,
        })
        
        if not patch_text:
            patch_text = ""  # Failed to generate valid patch
            
        return [{
            "patch_text": patch_text, 
            "summary": summary, 
            "metadata": {
                "model": model_name,
                "variant": variant_name,
                "attempt": len(attempt_history),
            }
        }]

    def update_attempt_result(success: bool, test_output: str = "") -> None:
        """Update the last attempt with its result and test output for learning."""
        nonlocal attempt_history
        
        if not attempt_history:
            return
            
        last = attempt_history[-1]
        last["result"] = "PASS" if success else "FAIL"
        last["test_output"] = test_output[:1000] if not success else ""  # Store test output for failures
        
        # Do not train prompt-bandit on forced variants (keeps separation: upstream policy vs local bandit)
        if not last.get("forced_variant", False):
            bandit.update(last["variant"], success=success)
            logger.info("Bandit updated: %s -> %s", last["variant"], "success" if success else "failure")
        else:
            logger.info("Bandit skipped update (forced variant): %s -> %s", last["variant"], "success" if success else "failure")

    def reset_attempt_history() -> None:
        """Reset attempt history for a new task."""
        nonlocal attempt_history
        attempt_history = []
        logger.debug("Attempt history reset")
        
    def get_bandit_stats() -> dict[str, Any]:
        """Get current bandit statistics."""
        return bandit.export_state()

    # Attach helper methods to the function
    _generate_patches.update_attempt_result = update_attempt_result  # type: ignore
    _generate_patches.reset_attempt_history = reset_attempt_history  # type: ignore
    _generate_patches.get_bandit_stats = get_bandit_stats  # type: ignore

    base_fn = _generate_patches

    if _trace_writer is not None:
        return make_recording_llm_wrapper(base_fn, _trace_writer)
    return base_fn


def get_active_trace_writer() -> Optional[TraceWriter]:
    return _trace_writer


def get_active_trace_reader() -> Optional[TraceReader]:
    return _trace_reader
