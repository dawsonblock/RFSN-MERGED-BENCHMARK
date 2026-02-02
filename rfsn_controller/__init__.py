# rfsn_controller package
from __future__ import annotations

"""
RFSN Sandbox Controller - Autonomous software engineering agent.

Core Modules:
    - cli: command-line entry point
    - controller: the main RFSN controller loop
    - config: configuration dataclasses (extracted from controller)
    - sandbox: utilities for managing disposable git sandboxes
    - verifier: test runner and result wrapper
    - parsers: helper functions for parsing test output
    - policy: heuristics for choosing repair intents and subgoals
    - prompt: helper for building model input strings
    - llm: LLM integrations (Gemini, DeepSeek, ensemble)
    - log: utility for writing JSONL logs

Optimization Modules:
    - docker_pool: warm container pool for faster Docker execution
    - semantic_cache: embedding-based cache for higher hit rates
    - prompt_compression: reduce token count for faster inference
    - streaming_validator: early termination for invalid responses
    - action_store: learn from past action outcomes
    - speculative_exec: predictive preloading
    - incremental_testing: run only affected tests

Planning Modules:
    - planner_v2: hierarchical task decomposition
    - cgw_bridge: Conscious Global Workspace integration

New in v0.3.0:
    - config: extracted configuration classes for better modularity
    - controller_helpers: extracted helper functions
    - config_from_cli_args: create config from CLI arguments
"""

# Convenience imports for common usage patterns
from .config import (
    BudgetConfig,
    ContractsConfig,
    ControllerConfig,
    config_from_cli_args,
)
from .run_id import make_run_id

__all__ = [
    "make_run_id",
    "BudgetConfig",
    "ContractsConfig",
    "ControllerConfig",
    "config_from_cli_args",
]

__version__ = "0.3.0"
