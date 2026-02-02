"""Controller context object passed throughout execution.

This module defines ControllerContext, a single object that holds all
runtime state and eliminates global configuration drift.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .budget import Budget
    from .config import ControllerConfig
    from .contracts import ContractRegistry, ContractValidator
    from .planner import PlanDAG
    from .policy_bandit import ThompsonBandit
    from .repo_index import RepoIndex


@dataclass
class EventLog:
    """Append-only structured event log."""
    
    path: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Ensure the output directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, event_type: str, **data: Any) -> None:
        """Emit a structured event to the log.
        
        Args:
            event_type: Type of event (e.g., "step_start", "patch_applied").
            **data: Additional event data.
        """
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": event_type,
            **data,
        }
        self.events.append(event)
        
        # Append to file immediately for durability
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    
    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get events, optionally filtered by type.
        
        Args:
            event_type: Optional type to filter by.
            
        Returns:
            List of matching events.
        """
        if event_type is None:
            return self.events.copy()
        return [e for e in self.events if e.get("type") == event_type]


@dataclass
class ControllerContext:
    """Central context object passed throughout controller execution.
    
    This eliminates global state and ensures all components have access
    to the same configuration and runtime objects.
    """
    
    config: ControllerConfig
    event_log: EventLog
    rng: random.Random = field(init=False)
    
    # Lazy-initialized components
    _sandbox: Any = field(default=None, repr=False)
    _repo_index: RepoIndex | None = field(default=None, repr=False)
    _plan: PlanDAG | None = field(default=None, repr=False)
    _policy: ThompsonBandit | None = field(default=None, repr=False)
    _budget: Budget | None = field(default=None, repr=False)
    _contract_registry: ContractRegistry | None = field(default=None, repr=False)
    _contract_validator: ContractValidator | None = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the seeded RNG."""
        self.rng = random.Random(self.config.seed)
        self.event_log.emit("context_initialized", seed=self.config.seed)
    
    @property
    def output_dir(self) -> Path:
        """Get the output directory path."""
        return Path(self.config.output_dir)
    
    @property
    def sandbox(self) -> Any:
        """Get the sandbox instance."""
        return self._sandbox
    
    @sandbox.setter
    def sandbox(self, value: Any) -> None:
        """Set the sandbox instance."""
        self._sandbox = value
    
    @property
    def repo_index(self) -> RepoIndex | None:
        """Get the repo index if enabled and built."""
        return self._repo_index
    
    @repo_index.setter
    def repo_index(self, value: RepoIndex) -> None:
        """Set the repo index."""
        self._repo_index = value
        self.event_log.emit("repo_index_set", files=len(value.files) if value else 0)
    
    @property
    def plan(self) -> PlanDAG | None:
        """Get the current execution plan."""
        return self._plan
    
    @plan.setter
    def plan(self, value: PlanDAG) -> None:
        """Set the execution plan."""
        self._plan = value
        self.event_log.emit("plan_set", nodes=len(value.nodes) if value else 0)
    
    @property
    def policy(self) -> ThompsonBandit | None:
        """Get the learning policy."""
        return self._policy
    
    @policy.setter
    def policy(self, value: ThompsonBandit) -> None:
        """Set the learning policy."""
        self._policy = value
        self.event_log.emit("policy_set", mode=self.config.policy_mode)
    
    @property
    def budget(self) -> Budget | None:
        """Get the budget tracker."""
        return self._budget
    
    @budget.setter
    def budget(self, value: Budget) -> None:
        """Set the budget tracker."""
        self._budget = value
        if value is not None:
            self.event_log.emit(
                "budget_set",
                max_steps=value.max_steps,
                max_llm_calls=value.max_llm_calls,
                max_tokens=value.max_tokens,
                max_time_seconds=value.max_time_seconds,
                max_subprocess_calls=value.max_subprocess_calls,
            )
    
    @property
    def contract_registry(self) -> ContractRegistry | None:
        """Get the contract registry."""
        return self._contract_registry
    
    @contract_registry.setter
    def contract_registry(self, value: ContractRegistry) -> None:
        """Set the contract registry."""
        self._contract_registry = value
        if value is not None:
            contracts = value.get_enabled()
            self.event_log.emit(
                "contract_registry_set",
                num_contracts=len(contracts),
                contracts=[c.name for c in contracts],
            )
    
    @property
    def contract_validator(self) -> ContractValidator | None:
        """Get the contract validator."""
        return self._contract_validator
    
    @contract_validator.setter
    def contract_validator(self, value: ContractValidator) -> None:
        """Set the contract validator."""
        self._contract_validator = value
        self.event_log.emit("contract_validator_set")
    
    def save_plan(self) -> str | None:
        """Save the current plan to disk.
        
        Returns:
            Path to the saved plan file, or None if no plan.
        """
        if self._plan is None:
            return None
        
        plan_path = self.output_dir / self.config.plan_file
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(self._plan.to_json(), f, indent=2)
        
        self.event_log.emit("plan_saved", path=str(plan_path))
        return str(plan_path)


def create_context(config: ControllerConfig) -> ControllerContext:
    """Create a new ControllerContext from configuration.
    
    Args:
        config: The controller configuration.
        
    Returns:
        Initialized ControllerContext.
    """
    from .budget import Budget, set_global_budget
    from .contracts import (
        ContractRegistry,
        ContractValidator,
        register_standard_contracts,
        set_global_registry,
        set_global_validator,
    )
    
    # Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create event log
    events_path = output_dir / config.events_file
    event_log = EventLog(path=events_path)
    
    # Create context
    ctx = ControllerContext(config=config, event_log=event_log)
    
    # Initialize budget from config if any limits are set
    budget_config = config.budget
    if any([
        budget_config.max_steps > 0,
        budget_config.max_llm_calls > 0,
        budget_config.max_tokens > 0,
        budget_config.max_time_seconds > 0,
        budget_config.max_subprocess_calls > 0,
    ]):
        budget = Budget(
            max_steps=budget_config.max_steps,
            max_llm_calls=budget_config.max_llm_calls,
            max_tokens=budget_config.max_tokens,
            max_time_seconds=budget_config.max_time_seconds,
            max_subprocess_calls=budget_config.max_subprocess_calls,
            warning_threshold=budget_config.warning_threshold,
        )
        ctx.budget = budget
        # Also set as global budget for modules that can't easily access context
        set_global_budget(budget)
    
    # Initialize contract system if enabled
    contracts_config = config.contracts
    if contracts_config.enabled:
        registry = ContractRegistry()
        
        # Register standard contracts based on config
        register_standard_contracts(registry)
        
        # Disable contracts based on config
        if not contracts_config.shell_execution_enabled:
            registry.disable("shell_execution")
        if not contracts_config.budget_tracking_enabled:
            registry.disable("budget_tracking")
        if not contracts_config.llm_calling_enabled:
            registry.disable("llm_calling")
        if not contracts_config.event_logging_enabled:
            registry.disable("event_logging")
        
        # Create validator
        validator = ContractValidator(registry)
        
        # Set on context
        ctx.contract_registry = registry
        ctx.contract_validator = validator
        
        # Also set as global for modules that can't easily access context
        set_global_registry(registry)
        set_global_validator(validator)
    
    return ctx
