import inspect


def test_llm_patch_generator_emits_apply_patch():
    """The LLM patch generator must emit APPLY_PATCH directly.

    This prevents silent no-op cycles where the runtime would accept a
    GENERATED_PATCH proposal but never apply it.
    """
    from cgw_ssl_guard.coding_agent.llm_integration import LLMPatchGenerator

    src = inspect.getsource(LLMPatchGenerator)
    assert "CodingAction.APPLY_PATCH" in src
    assert "CodingAction.GENERATE_PATCH" not in src
