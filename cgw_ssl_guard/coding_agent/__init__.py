"""Coding agent module for serial decision architecture.

This module provides a production-grade autonomous coding agent built
on the CGW (Conscious Global Workspace) / SSL guard architecture.

Key Features:
- Serial decision loop (exactly one decision per cycle)
- Thalamic gate arbitration with forced overrides
- Blocking execution (no tool overlap)
- Event emission for replay and auditing
- Seriality monitoring for invariant verification

Usage:
    from cgw_ssl_guard.coding_agent import CodingAgentRuntime, AgentConfig
    
    config = AgentConfig(goal="Fix failing tests", max_cycles=50)
    runtime = CodingAgentRuntime(config=config)
    result = runtime.run_until_done()
    
    if result.success:
        print(f"Fixed in {result.cycles_executed} cycles")
    else:
        print(f"Failed: {result.error}")

Architecture:
    Decision Layer (CGW + Gate)
    │
    ├── Proposal Generators → Candidates
    │   ├── SafetyProposalGenerator (ABORT on safety trigger)
    │   ├── PlannerProposalGenerator (next step from goal)
    │   ├── MemoryProposalGenerator (historical patterns)
    │   └── IdleProposalGenerator (fallback)
    │
    ├── Thalamic Gate → Single Winner Selection
    │   └── Forced queue checked first
    │
    ├── CGW Runtime → Atomic Commit
    │   └── One slot, atomic swap
    │
    └── Blocking Executor → Action Execution
        └── Runs tests, applies patches, etc.
        └── Returns results for next cycle
"""

from .action_types import (
    ActionCategory,
    ActionPayload,
    CodingAction,
    CycleResult,
    ExecutionResult,
    ACTION_CATEGORIES,
)

from .proposal_generators import (
    AnalyzerProposalGenerator,
    IdleProposalGenerator,
    MemoryProposalGenerator,
    PlannerProposalGenerator,
    ProposalContext,
    ProposalGenerator,
    SafetyProposalGenerator,
)

from .executor import (
    BlockingExecutor,
    ExecutorConfig,
    SandboxProtocol,
)

from .coding_agent_runtime import (
    AgentConfig,
    AgentResult,
    CodingAgentRuntime,
)


# Optional: Replay module (can be removed for minimal install)
try:
    from .replay import (
        EventReplayEngine,
        ReplayCycle,
        ReplayEvent,
        SessionAnalysis,
    )
    HAS_REPLAY = True
except ImportError:
    HAS_REPLAY = False

from .llm_integration import (
    LLMAnalysisGenerator,
    LLMConfig,
    LLMDecisionAdvisor,
    LLMPatchGenerator,
)

from .llm_adapter import (
    create_llm_caller,
    get_default_router,
    LLMAdapterConfig,
    LLMRouter,
    validate_api_keys,
)

from .docker_sandbox import (
    check_docker_available,
    create_executor_sandbox,
    DockerSandbox,
    DockerSandboxConfig,
    SandboxManager,
)

# Optional: CGW Metrics (can be removed for minimal install)
try:
    from .cgw_metrics import (
        CGWMetricsCollector,
        get_metrics_collector,
        start_dashboard_server,
    )
    HAS_CGW_METRICS = True
except ImportError:
    HAS_CGW_METRICS = False

from .cgw_bandit import (
    BanditBoostMixin,
    CGWBandit,
    CGWBanditConfig,
    get_cgw_bandit,
    record_action_outcome,
)

from .event_store import (
    CGWEventStore,
    EventStoreConfig,
    EventStoreSubscriber,
    get_event_store,
    StoredEvent,
)

from .streaming_llm import (
    create_streaming_caller,
    StreamingConfig,
    StreamingLLMClient,
    StreamingMetrics,
    SyncStreamingWrapper,
)

from .action_memory import (
    CGWActionMemory,
    CGWMemoryConfig,
    get_action_memory,
    MemoryExecutorMixin,
)

# Optional: WebSocket Dashboard (can be removed for minimal install)
try:
    from .websocket_dashboard import (
        CGWDashboardServer,
        DashboardConfig,
        DashboardEventSubscriber,
        get_dashboard,
    )
    HAS_WEBSOCKET_DASHBOARD = True
except ImportError:
    HAS_WEBSOCKET_DASHBOARD = False

from .config import (
    CGWConfig,
    load_config,
    create_default_config,
)

from .integrated_runtime import (
    IntegratedCGWAgent,
    run_agent,
)

__all__ = [


    # Action types
    "ActionCategory",
    "ActionPayload",
    "CodingAction",
    "CycleResult",
    "ExecutionResult",
    "ACTION_CATEGORIES",
    
    # Proposal generators
    "AnalyzerProposalGenerator",
    "IdleProposalGenerator",
    "MemoryProposalGenerator",
    "PlannerProposalGenerator",
    "ProposalContext",
    "ProposalGenerator",
    "SafetyProposalGenerator",
    
    # Executor
    "BlockingExecutor",
    "ExecutorConfig",
    "SandboxProtocol",
    
    # Runtime
    "AgentConfig",
    "AgentResult",
    "CodingAgentRuntime",
    
    # Replay
    "EventReplayEngine",
    "ReplayCycle",
    "ReplayEvent",
    "SessionAnalysis",
    
    # LLM Integration
    "LLMAnalysisGenerator",
    "LLMConfig",
    "LLMDecisionAdvisor",
    "LLMPatchGenerator",
    
    # LLM Adapter (real API integration)
    "create_llm_caller",
    "get_default_router",
    "LLMAdapterConfig",
    "LLMRouter",
    "validate_api_keys",
    
    # Docker Sandbox
    "check_docker_available",
    "create_executor_sandbox",
    "DockerSandbox",
    "DockerSandboxConfig",
    "SandboxManager",
    
    # Metrics
    "CGWMetricsCollector",
    "get_metrics_collector",
    "start_dashboard_server",
    
    # Strategy Bandit (Phase 2)
    "BanditBoostMixin",
    "CGWBandit",
    "CGWBanditConfig",
    "get_cgw_bandit",
    "record_action_outcome",
    
    # Event Store (Phase 2)
    "CGWEventStore",
    "EventStoreConfig",
    "EventStoreSubscriber",
    "get_event_store",
    "StoredEvent",
    
    # Streaming LLM (Phase 2)
    "create_streaming_caller",
    "StreamingConfig",
    "StreamingLLMClient",
    "StreamingMetrics",
    "SyncStreamingWrapper",
    
    # Action Memory (Phase 2)
    "CGWActionMemory",
    "CGWMemoryConfig",
    "get_action_memory",
    "MemoryExecutorMixin",
    
    # WebSocket Dashboard (Phase 2)
    "CGWDashboardServer",
    "DashboardConfig",
    "DashboardEventSubscriber",
    "get_dashboard",
    
    # Configuration (Phase 3)
    "CGWConfig",
    "load_config",
    "create_default_config",
    
    # Integrated Runtime (Phase 3)
    "IntegratedCGWAgent",
    "run_agent",
]
