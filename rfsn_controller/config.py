"""Controller configuration dataclasses.

This module contains configuration dataclasses extracted from controller.py
for better modularity and reusability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetConfig:
    """Budget configuration for resource limits."""
    
    max_steps: int = 0
    max_llm_calls: int = 0
    max_tokens: int = 0
    max_time_seconds: float = 0
    max_subprocess_calls: int = 0
    warning_threshold: float = 0.8


@dataclass
class ContractsConfig:
    """Contracts configuration for runtime safety checks."""
    
    enabled: bool = True
    shell_execution_enabled: bool = True
    budget_tracking_enabled: bool = True
    llm_calling_enabled: bool = True
    event_logging_enabled: bool = True


@dataclass
class EventConfig:
    """Event system configuration."""
    
    enabled: bool = True
    storage_path: str = ".rfsn/events.jsonl"
    max_events_memory: int = 10000
    persist_events: bool = True


@dataclass
class ControllerConfig:
    """Configuration for a controller run."""

    github_url: str
    test_cmd: str = "pytest -q"
    ref: str | None = None
    max_steps: int = 12
    temps: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4])
    fix_all: bool = False
    max_steps_without_progress: int = 10
    collect_finetuning_data: bool = False
    model: str = "deepseek-chat"
    max_minutes: int = 30
    install_timeout: int = 300
    focus_timeout: int = 120
    full_timeout: int = 300
    max_tool_calls: int = 40
    docker_image: str = "python:3.11-slim"
    unsafe_host_exec: bool = False
    cpu: float = 2.0
    mem_mb: int = 4096
    pids: int = 256
    docker_readonly: bool = False
    lint_cmd: str | None = None
    typecheck_cmd: str | None = None
    repro_cmd: str | None = None
    verify_cmd: str | None = None
    dry_run: bool = False
    project_type: str = "auto"
    buildpack: str = "auto"
    enable_sysdeps: bool = False
    sysdeps_tier: int = 4
    sysdeps_max_packages: int = 10
    build_cmd: str | None = None
    learning_db_path: str | None = None
    learning_half_life_days: int = 14
    learning_max_age_days: int = 90
    learning_max_rows: int = 20000
    time_mode: str = "frozen"  # frozen|live
    run_started_at_utc: str | None = None
    time_seed: int | None = None
    rng_seed: int | None = None
    feature_mode: bool = False
    feature_description: str | None = None
    acceptance_criteria: list[str] = field(default_factory=list)
    # Verification configuration for feature mode
    verify_policy: str = "tests_only"  # tests_only | cmds_then_tests | cmds_only
    focused_verify_cmds: list[str] = field(default_factory=list)
    verify_cmds: list[str] = field(default_factory=list)
    # Hygiene configuration overrides
    max_lines_changed: int | None = None
    max_files_changed: int | None = None
    allow_lockfile_changes: bool = False
    # Phase budget limits for reliability
    max_install_attempts: int = 3
    max_patch_attempts: int = 20
    max_verification_attempts: int = 5
    # Verification repeatability
    repro_times: int = 1  # Run verification N times to ensure reproducibility
    # Performance optimizations
    enable_llm_cache: bool = False  # Enable LLM response caching
    llm_cache_path: str | None = None  # Path to LLM cache database
    parallel_patches: bool = True  # Generate patches in parallel (faster)
    ensemble_mode: bool = False  # Use multi-model ensemble
    incremental_tests: bool = False  # Run only affected tests first
    enable_telemetry: bool = False  # Enable OpenTelemetry/Prometheus
    telemetry_port: int = 8080  # Prometheus metrics port
    # Elite Controller options
    policy_mode: str = "off"  # off | bandit
    planner_mode: str = "off"  # off | dag | v2 | v5
    repo_index: bool = False  # Enable repo indexing
    seed: int = 1337  # Deterministic seed
    # Risk & persistence
    risk_profile: str = "production"  # production | research
    state_dir: str | None = None  # base host dir; we create <base>/<risk>/<run_id>/
    # Verification durability
    durability_reruns: int = 0  # rerun full tests N additional times after success
    no_eval: bool = False  # Skip final evaluation
    # Context-related configuration (for create_context compatibility)
    output_dir: str = ".rfsn"  # Output directory for artifacts
    events_file: str = "events.jsonl"  # Events log filename
    plan_file: str = "plan.json"  # Plan filename
    # Budget configuration (inline for context compatibility)
    budget: BudgetConfig = field(default_factory=lambda: BudgetConfig())
    # Contracts configuration (inline for context compatibility)
    contracts: ContractsConfig = field(default_factory=lambda: ContractsConfig())
    # Beam search configuration
    beam_search_enabled: bool = False  # Enable multi-step beam search
    beam_width: int = 3  # Number of candidates to keep per step
    beam_depth: int = 5  # Maximum expansion depth
    beam_score_threshold: float = 0.95  # Score to terminate early
    beam_timeout_seconds: float = 300.0  # Total search timeout
    # Events configuration
    events: EventConfig = field(default_factory=lambda: EventConfig())


def config_from_cli_args(args) -> ControllerConfig:  # noqa: PLR0912
    """Create a ControllerConfig from CLI arguments.
    
    Args:
        args: Parsed command line arguments (argparse.Namespace or similar).
        
    Returns:
        ControllerConfig instance populated from CLI args.
    """
    # Map CLI argument names to config field names
    config_kwargs = {}
    
    # Required field
    if hasattr(args, 'github_url'):
        config_kwargs['github_url'] = args.github_url
    elif hasattr(args, 'repo'):
        config_kwargs['github_url'] = args.repo
    else:
        raise ValueError("github_url or repo argument is required")
    
    # Map common CLI fields to config fields
    field_mappings = {
        'test_cmd': 'test_cmd',
        'ref': 'ref',
        'max_steps': 'max_steps',
        'model': 'model',
        'max_minutes': 'max_minutes',
        'docker_image': 'docker_image',
        'dry_run': 'dry_run',
        'fix_all': 'fix_all',
        'buildpack': 'buildpack',
        'seed': 'seed',
        'policy_mode': 'policy_mode',
        'planner_mode': 'planner_mode',
        'enable_telemetry': 'enable_telemetry',
        'telemetry_port': 'telemetry_port',
        'output_dir': 'output_dir',
    }
    
    for cli_name, config_name in field_mappings.items():
        if hasattr(args, cli_name):
            value = getattr(args, cli_name)
            if value is not None:
                config_kwargs[config_name] = value
    
    # Handle budget configuration from CLI args
    budget_kwargs = {}
    budget_mappings = {
        'budget_max_steps': 'max_steps',
        'budget_max_llm_calls': 'max_llm_calls',
        'budget_max_tokens': 'max_tokens',
        'budget_max_time_seconds': 'max_time_seconds',
        'budget_max_subprocess_calls': 'max_subprocess_calls',
        'budget_warning_threshold': 'warning_threshold',
    }
    
    for cli_name, budget_name in budget_mappings.items():
        if hasattr(args, cli_name):
            value = getattr(args, cli_name)
            if value is not None:
                budget_kwargs[budget_name] = value
    
    if budget_kwargs:
        config_kwargs['budget'] = BudgetConfig(**budget_kwargs)
    
    # Handle events configuration from CLI args
    events_kwargs = {}
    events_mappings = {
        'events_enabled': 'enabled',
        'events_storage_path': 'storage_path',
        'events_max_memory': 'max_events_memory',
        'events_persist': 'persist_events',
    }
    
    for cli_name, events_name in events_mappings.items():
        if hasattr(args, cli_name):
            value = getattr(args, cli_name)
            if value is not None:
                events_kwargs[events_name] = value
    
    if events_kwargs:
        config_kwargs['events'] = EventConfig(**events_kwargs)
    
    return ControllerConfig(**config_kwargs)
