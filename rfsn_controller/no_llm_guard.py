import os

def assert_llm_allowed():
    if os.environ.get("RFSN_DISALLOW_LLM") == "1":
        raise RuntimeError("LLM calls disabled by RFSN_DISALLOW_LLM=1 (replay/verification mode)")
