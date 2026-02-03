import os
from typing import Any, Optional
from rfsn_controller.replay_trace import (
    TraceWriter, TraceReader,
    make_recording_llm_wrapper, make_replay_llm_wrapper
)

_RFSN_RECORD_TRACE = os.environ.get("RFSN_RECORD_TRACE")
_RFSN_REPLAY_TRACE = os.environ.get("RFSN_REPLAY_TRACE")

_trace_writer = TraceWriter(_RFSN_RECORD_TRACE, run_meta={"mode": "record"}) if _RFSN_RECORD_TRACE else None
_trace_reader = TraceReader(_RFSN_REPLAY_TRACE, verify_chain=True) if _RFSN_REPLAY_TRACE else None

def get_llm_patch_fn(model_name: str = "deepseek"):
    """Factory for patch-candidate generator. Replay avoids any LLM imports/calls."""
    if _trace_reader is not None:
        return make_replay_llm_wrapper(_trace_reader)

    # Lazy import so replay/verification can import this module with LLM disabled.
    from rfsn_upstream.prompt_variants import get_variant, format_prompt
    from rfsn_upstream.llm_prompting import extract_diff_from_response, call_llm, LLMConfig, LLMProvider
    

    def _generate_patches(plan: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Default patch generator using upstream prompt variants."""
        # 1. Select variant (could be bandit-selected in future)
        variant_name = "v_diagnose_then_patch"
        variant = get_variant(variant_name)
        
        # 2. Build prompt
        task_info = plan.task if hasattr(plan, "task") else {} # Handle plan object or dict
        if not task_info and isinstance(plan, dict):
             task_info = plan.get("task", {})

        problem_statement = task_info.get("problem_statement", "")
        # Get file content if available in context or plan
        # Simplification: assume plan has file context or we rely on what's in 'context'
        # context['retrieval'] might have useful stuff
        
        # For now, let's construct a simple context from what we have
        # Note: The context passed from propose_v2 has hypotheses, retrieval, etc.
        
        # Prepare file content structure for the prompt
        # Note: content already has line numbers prepended by propose_v2
        raw_files = context.get("file_context", {})
        file_content_str = ""
        for path, content in raw_files.items():
            file_content_str += f"\nFile: {path} (line numbers shown on left)\n```python\n{content}\n```\n"

        system_prompt, user_prompt = format_prompt(
            variant,
            problem_statement=problem_statement,
            test_output=task_info.get("last_test_output", ""),
            file_content=file_content_str,
        )
        # 3. Call LLM
        config = LLMConfig(provider=LLMProvider.DEEPSEEK, temperature=variant.temperature)
        response = call_llm(user_prompt, system_prompt=system_prompt, config=config)

        
        content = response.content
        summary = "Generated patch"
        if response.parsed and isinstance(response.parsed, dict):
             summary = response.parsed.get("summary", summary)
        
        # 4. Extract Diff
        patch_text = extract_diff_from_response(content)
        
        # DEBUG: Log what we're extracting
        import logging
        debug_logger = logging.getLogger("llm_patcher")
        debug_logger.info("=" * 60)
        debug_logger.info("LLM RESPONSE (first 2000 chars):")
        debug_logger.info(content[:2000] if content else "<empty>")
        debug_logger.info("=" * 60)
        debug_logger.info("EXTRACTED PATCH (first 1000 chars):")
        debug_logger.info(patch_text[:1000] if patch_text else "<none>")
        debug_logger.info("=" * 60)
        
        if not patch_text:
            patch_text = "" # Failed to generate valid patch
            
        return [{"patch_text": patch_text, "summary": summary, "metadata": {"model": model_name}}]

    base_fn = _generate_patches

    if _trace_writer is not None:
        return make_recording_llm_wrapper(base_fn, _trace_writer)
    return base_fn

def get_active_trace_writer() -> Optional[TraceWriter]:
    return _trace_writer

def get_active_trace_reader() -> Optional[TraceReader]:
    return _trace_reader
