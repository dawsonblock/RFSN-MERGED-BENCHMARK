"""RFSN Upstream Learner.

The upstream learner lives OUTSIDE the kernel and:
- Selects prompt variants (bandit)
- Fingerprints failures for retrieval
- Stores memories for context
- Updates based on task outcomes
- Runs SWE-bench training loops

INVARIANTS:
1. Learner NEVER modifies kernel decisions
2. Learner ONLY affects proposal generation (coaching layer)
3. Learner state is persisted for CI/CD integration
"""

from .bandit import ThompsonBandit, ArmStats
from .fingerprint import Fingerprint, compute_fingerprint, fingerprint_from_rejection
from .retrieval import Memory, MemoryIndex
from .prompt_variants import PROMPT_VARIANTS, get_variant, PromptVariant

# SWE-bench specific modules
from .worktree_manager import WorktreeManager, WorktreeHandle
from .reward import (
    TaskOutcome,
    OutcomeType,
    RewardConfig,
    compute_reward,
    create_success_outcome,
    create_failure_outcome,
    create_rejection_outcome,
)
from .critic import PlannerCritic, Critique, CritiqueIssue, CriticConfig
from .llm_prompting import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    call_llm,
    parse_json_response,
    extract_diff_from_response,
)

__version__ = "1.1.0"
__all__ = [
    # Bandit
    "ThompsonBandit",
    "ArmStats",
    # Fingerprinting
    "Fingerprint",
    "compute_fingerprint",
    "fingerprint_from_rejection",
    # Retrieval
    "Memory",
    "MemoryIndex",
    # Prompts
    "PROMPT_VARIANTS",
    "get_variant",
    "PromptVariant",
    # Worktree management
    "WorktreeManager",
    "WorktreeHandle",
    # Reward
    "TaskOutcome",
    "OutcomeType",
    "RewardConfig",
    "compute_reward",
    "create_success_outcome",
    "create_failure_outcome",
    "create_rejection_outcome",
    # Critic
    "PlannerCritic",
    "Critique",
    "CritiqueIssue",
    "CriticConfig",
    # LLM
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "call_llm",
    "parse_json_response",
    "extract_diff_from_response",
]

