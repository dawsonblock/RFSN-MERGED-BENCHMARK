"""Feature contracts for capability management and validation.

This module provides a contract system for defining and enforcing agent capabilities:
- FeatureContract: Dataclass defining agent capability requirements
- ContractViolation: Exception for contract enforcement failures
- ContractRegistry: Central registry for discovering and managing features
- ContractValidator: Validates operations against registered contracts

Contracts provide clear capability boundaries and ensure proper validation
without adding excessive runtime overhead.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Contract Types and Constraints
# =============================================================================

class ContractConstraint(Enum):
    """Standard constraints that can be applied to contracts."""
    
    NO_SHELL_TRUE = "no_shell_true"  # Forbid shell=True in subprocess calls
    NO_INTERACTIVE_SHELL = "no_interactive_shell"  # Forbid interactive shells
    NO_SHELL_WRAPPERS = "no_shell_wrappers"  # Forbid sh -c / bash -c patterns
    ENFORCE_BUDGET_LIMITS = "enforce_budget_limits"  # Require budget tracking
    REQUIRE_ALLOWLIST = "require_allowlist"  # Require command allowlist check
    LOG_ALL_OPERATIONS = "log_all_operations"  # Log all operations to events
    TRACK_TOKEN_USAGE = "track_token_usage"  # Track LLM token consumption
    VALIDATE_INPUTS = "validate_inputs"  # Validate inputs before operations


# =============================================================================
# Contract Violation Exception
# =============================================================================

class ContractViolation(Exception):
    """Exception raised when a contract constraint is violated.
    
    This exception includes detailed information about the violation
    for debugging and audit purposes.
    """
    
    def __init__(
        self,
        contract_name: str,
        constraint: ContractConstraint | str,
        operation: str,
        details: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize contract violation.
        
        Args:
            contract_name: Name of the violated contract.
            constraint: The specific constraint that was violated.
            operation: The operation that caused the violation.
            details: Human-readable details about the violation.
            context: Additional context data for debugging.
        """
        self.contract_name = contract_name
        self.constraint = constraint if isinstance(constraint, str) else constraint.value
        self.operation = operation
        self.details = details
        self.context = context or {}
        self.timestamp = datetime.now(UTC).isoformat()
        
        message = f"Contract '{contract_name}' violated: {self.constraint}"
        if details:
            message += f" - {details}"
        if operation:
            message += f" (operation: {operation})"
        
        super().__init__(message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary for logging/serialization."""
        return {
            "contract_name": self.contract_name,
            "constraint": self.constraint,
            "operation": self.operation,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp,
        }
    
    def __repr__(self) -> str:
        return (
            f"ContractViolation(contract={self.contract_name!r}, "
            f"constraint={self.constraint!r}, operation={self.operation!r})"
        )


# =============================================================================
# Feature Contract Dataclass
# =============================================================================

@dataclass
class FeatureContract:
    """Defines agent capability requirements and constraints.
    
    A feature contract specifies what tools and constraints are required
    for a particular feature to operate correctly. This allows for:
    - Capability discovery and dependency checking
    - Runtime validation of operations
    - Clear documentation of feature requirements
    
    Attributes:
        name: Unique identifier for this contract.
        version: Semantic version of the contract (e.g., "1.0.0").
        description: Human-readable description of the feature.
        required_tools: Tools that must be available for this feature.
        optional_tools: Tools that enhance the feature but aren't required.
        constraints: Set of constraints this feature enforces.
        enabled: Whether this contract is currently active.
        metadata: Additional metadata for the contract.
    """
    
    name: str
    version: str
    description: str
    required_tools: set[str] = field(default_factory=set)
    optional_tools: set[str] = field(default_factory=set)
    constraints: set[ContractConstraint] = field(default_factory=set)
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate contract after initialization."""
        if not self.name:
            raise ValueError("Contract name cannot be empty")
        if not self.version:
            raise ValueError("Contract version cannot be empty")
        
        # Ensure sets are proper types
        if isinstance(self.required_tools, (list, tuple)):
            self.required_tools = set(self.required_tools)
        if isinstance(self.optional_tools, (list, tuple)):
            self.optional_tools = set(self.optional_tools)
        if isinstance(self.constraints, (list, tuple)):
            self.constraints = set(self.constraints)
    
    def has_constraint(self, constraint: ContractConstraint) -> bool:
        """Check if contract has a specific constraint."""
        return constraint in self.constraints
    
    def requires_tool(self, tool: str) -> bool:
        """Check if contract requires a specific tool."""
        return tool in self.required_tools
    
    def uses_tool(self, tool: str) -> bool:
        """Check if contract uses a tool (required or optional)."""
        return tool in self.required_tools or tool in self.optional_tools
    
    def to_dict(self) -> dict[str, Any]:
        """Convert contract to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "constraints": [c.value for c in self.constraints],
            "enabled": self.enabled,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureContract:
        """Create contract from dictionary."""
        constraints = set()
        for c in data.get("constraints", []):
            try:
                constraints.add(ContractConstraint(c))
            except ValueError:
                logger.warning(f"Unknown constraint: {c}")
        
        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            required_tools=set(data.get("required_tools", [])),
            optional_tools=set(data.get("optional_tools", [])),
            constraints=constraints,
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Contract Registry
# =============================================================================

class ContractRegistry:
    """Central registry for feature contracts.
    
    The registry provides:
    - Contract registration and discovery
    - Dependency checking
    - Event logging for contract changes
    - Thread-safe operations
    """
    
    def __init__(self) -> None:
        """Initialize the contract registry."""
        self._contracts: dict[str, FeatureContract] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[str, FeatureContract], None]] = []
    
    def register(
        self,
        contract: FeatureContract,
        replace: bool = False,
    ) -> bool:
        """Register a feature contract.
        
        Args:
            contract: The contract to register.
            replace: If True, replace existing contract with same name.
            
        Returns:
            True if registration succeeded, False if contract exists.
            
        Raises:
            ValueError: If contract is invalid.
        """
        with self._lock:
            if contract.name in self._contracts and not replace:
                logger.warning(f"Contract already registered: {contract.name}")
                return False
            
            self._contracts[contract.name] = contract
            logger.debug(f"Registered contract: {contract.name} v{contract.version}")
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    listener("registered", contract)
                except Exception as e:
                    logger.error(f"Contract listener error: {e}")
            
            # Log to event system if available
            self._log_registration(contract)
            
            return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a contract by name.
        
        Args:
            name: The contract name to unregister.
            
        Returns:
            True if unregistered, False if not found.
        """
        with self._lock:
            if name not in self._contracts:
                return False
            
            contract = self._contracts.pop(name)
            logger.debug(f"Unregistered contract: {name}")
            
            for listener in self._listeners:
                try:
                    listener("unregistered", contract)
                except Exception as e:
                    logger.error(f"Contract listener error: {e}")
            
            return True
    
    def get(self, name: str) -> FeatureContract | None:
        """Get a contract by name."""
        with self._lock:
            return self._contracts.get(name)
    
    def get_enabled(self) -> list[FeatureContract]:
        """Get all enabled contracts."""
        with self._lock:
            return [c for c in self._contracts.values() if c.enabled]
    
    def get_all(self) -> list[FeatureContract]:
        """Get all registered contracts."""
        with self._lock:
            return list(self._contracts.values())
    
    def get_by_constraint(self, constraint: ContractConstraint) -> list[FeatureContract]:
        """Get contracts that have a specific constraint."""
        with self._lock:
            return [
                c for c in self._contracts.values()
                if c.enabled and c.has_constraint(constraint)
            ]
    
    def get_by_tool(self, tool: str) -> list[FeatureContract]:
        """Get contracts that use a specific tool."""
        with self._lock:
            return [
                c for c in self._contracts.values()
                if c.enabled and c.uses_tool(tool)
            ]
    
    def has_contract(self, name: str) -> bool:
        """Check if a contract is registered."""
        with self._lock:
            return name in self._contracts
    
    def is_enabled(self, name: str) -> bool:
        """Check if a contract is registered and enabled."""
        with self._lock:
            contract = self._contracts.get(name)
            return contract is not None and contract.enabled
    
    def enable(self, name: str) -> bool:
        """Enable a contract."""
        with self._lock:
            contract = self._contracts.get(name)
            if contract is None:
                return False
            contract.enabled = True
            return True
    
    def disable(self, name: str) -> bool:
        """Disable a contract."""
        with self._lock:
            contract = self._contracts.get(name)
            if contract is None:
                return False
            contract.enabled = False
            return True
    
    def add_listener(
        self,
        listener: Callable[[str, FeatureContract], None],
    ) -> None:
        """Add a listener for contract registration events."""
        with self._lock:
            self._listeners.append(listener)
    
    def remove_listener(
        self,
        listener: Callable[[str, FeatureContract], None],
    ) -> None:
        """Remove a contract listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def check_dependencies(
        self,
        contract: FeatureContract,
        available_tools: set[str],
    ) -> list[str]:
        """Check if all required tools are available.
        
        Args:
            contract: The contract to check.
            available_tools: Set of available tool names.
            
        Returns:
            List of missing required tools (empty if all satisfied).
        """
        missing = contract.required_tools - available_tools
        return list(missing)
    
    def clear(self) -> None:
        """Clear all registered contracts."""
        with self._lock:
            self._contracts.clear()
    
    def _log_registration(self, contract: FeatureContract) -> None:
        """Log contract registration to event system."""
        try:
            from .events import log_feature_registered_global
            log_feature_registered_global(
                contract.name,
                contract.version,
                list(contract.required_tools),
            )
        except ImportError:
            pass  # Events module not available


# =============================================================================
# Contract Validator
# =============================================================================

class ContractValidator:
    """Validates operations against registered contracts.
    
    The validator checks operations against contract constraints
    and provides clear error messages on violations.
    """
    
    def __init__(self, registry: ContractRegistry) -> None:
        """Initialize validator with a contract registry.
        
        Args:
            registry: The contract registry to validate against.
        """
        self._registry = registry
        self._violation_handlers: list[Callable[[ContractViolation], None]] = []
    
    @property
    def registry(self) -> ContractRegistry:
        """Get the associated registry."""
        return self._registry
    
    def add_violation_handler(
        self,
        handler: Callable[[ContractViolation], None],
    ) -> None:
        """Add a handler for contract violations."""
        self._violation_handlers.append(handler)
    
    def remove_violation_handler(
        self,
        handler: Callable[[ContractViolation], None],
    ) -> None:
        """Remove a violation handler."""
        if handler in self._violation_handlers:
            self._violation_handlers.remove(handler)
    
    def _handle_violation(self, violation: ContractViolation) -> None:
        """Handle a contract violation."""
        # Log the violation
        logger.warning(str(violation))
        
        # Log to event system
        self._log_violation_event(violation)
        
        # Notify handlers
        for handler in self._violation_handlers:
            try:
                handler(violation)
            except Exception as e:
                logger.error(f"Violation handler error: {e}")
    
    def _log_violation_event(self, violation: ContractViolation) -> None:
        """Log violation to event system."""
        try:
            from .events import log_security_violation_global
            # Use the events module's expected signature
            log_security_violation_global(
                violation_type=f"contract_violation:{violation.constraint}",
                file_path=violation.contract_name,
                line_number=0,
                message=violation.details or violation.operation,
                severity="high",
            )
        except (ImportError, Exception):
            # If events module not available or call fails, just log warning
            logger.debug(f"Could not log violation event: {violation}")
    
    def validate_shell_execution(
        self,
        argv: list[str],
        shell: bool = False,
        operation: str = "shell_execution",
    ) -> None:
        """Validate shell execution against contracts.
        
        Args:
            argv: Command arguments.
            shell: Whether shell=True was requested.
            operation: Description of the operation.
            
        Raises:
            ContractViolation: If any shell constraint is violated.
        """
        import os
        
        # Get contracts with shell constraints
        contracts = self._registry.get_by_constraint(ContractConstraint.NO_SHELL_TRUE)
        
        for contract in contracts:
            if shell:
                violation = ContractViolation(
                    contract_name=contract.name,
                    constraint=ContractConstraint.NO_SHELL_TRUE,
                    operation=operation,
                    details="shell=True is forbidden",
                    context={"argv": argv[:3] if argv else []},
                )
                self._handle_violation(violation)
                raise violation
        
        # Check for shell wrappers
        contracts = self._registry.get_by_constraint(ContractConstraint.NO_SHELL_WRAPPERS)
        
        if argv and len(argv) >= 2:
            base_cmd = os.path.basename(argv[0])
            if base_cmd in ("sh", "bash", "dash", "zsh", "ksh") and "-c" in argv:
                for contract in contracts:
                    violation = ContractViolation(
                        contract_name=contract.name,
                        constraint=ContractConstraint.NO_SHELL_WRAPPERS,
                        operation=operation,
                        details=f"Shell wrapper detected: {base_cmd} -c",
                        context={"command": " ".join(argv[:3])},
                    )
                    self._handle_violation(violation)
                    raise violation
        
        # Check for interactive shells
        contracts = self._registry.get_by_constraint(
            ContractConstraint.NO_INTERACTIVE_SHELL
        )
        
        if argv and len(argv) >= 1:
            base_cmd = os.path.basename(argv[0])
            if base_cmd in ("sh", "bash", "dash", "zsh", "ksh"):
                if "-i" in argv or (len(argv) == 1):
                    for contract in contracts:
                        violation = ContractViolation(
                            contract_name=contract.name,
                            constraint=ContractConstraint.NO_INTERACTIVE_SHELL,
                            operation=operation,
                            details=f"Interactive shell detected: {base_cmd}",
                            context={"argv": argv},
                        )
                        self._handle_violation(violation)
                        raise violation
    
    def validate_budget_operation(
        self,
        resource: str,
        current: int,
        limit: int,
        operation: str = "budget_operation",
    ) -> None:
        """Validate budget operation against contracts.
        
        Args:
            resource: The resource being consumed.
            current: Current usage.
            limit: The limit for this resource.
            operation: Description of the operation.
            
        Raises:
            ContractViolation: If budget limits are violated.
        """
        contracts = self._registry.get_by_constraint(
            ContractConstraint.ENFORCE_BUDGET_LIMITS
        )
        
        if limit > 0 and current >= limit:
            for contract in contracts:
                violation = ContractViolation(
                    contract_name=contract.name,
                    constraint=ContractConstraint.ENFORCE_BUDGET_LIMITS,
                    operation=operation,
                    details=f"Budget exceeded for {resource}: {current}/{limit}",
                    context={"resource": resource, "current": current, "limit": limit},
                )
                self._handle_violation(violation)
                raise violation
    
    def validate_operation(
        self,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Generic operation validation against all applicable contracts.
        
        This method can be extended to validate various operation types.
        
        Args:
            operation: The operation being performed.
            context: Additional context for validation.
        """
        context = context or {}
        
        # Validate based on operation type
        if operation.startswith("shell:"):
            argv = context.get("argv", [])
            shell = context.get("shell", False)
            self.validate_shell_execution(argv, shell, operation)
        
        elif operation.startswith("budget:"):
            resource = context.get("resource", "unknown")
            current = context.get("current", 0)
            limit = context.get("limit", 0)
            self.validate_budget_operation(resource, current, limit, operation)
    
    def is_operation_allowed(
        self,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if an operation is allowed without raising.
        
        Args:
            operation: The operation to check.
            context: Additional context for validation.
            
        Returns:
            True if allowed, False if would violate a contract.
        """
        try:
            self.validate_operation(operation, context)
            return True
        except ContractViolation:
            return False


# =============================================================================
# Standard Contracts
# =============================================================================

def create_shell_execution_contract() -> FeatureContract:
    """Create the standard shell execution contract.
    
    This contract enforces safe subprocess execution:
    - No shell=True
    - No shell wrappers (sh -c, bash -c)
    - No interactive shells
    - Require allowlist checking
    """
    return FeatureContract(
        name="shell_execution",
        version="1.0.0",
        description="Secure shell/subprocess execution with safety constraints",
        required_tools={"safe_run", "subprocess"},
        optional_tools={"docker_exec"},
        constraints={
            ContractConstraint.NO_SHELL_TRUE,
            ContractConstraint.NO_SHELL_WRAPPERS,
            ContractConstraint.NO_INTERACTIVE_SHELL,
            ContractConstraint.REQUIRE_ALLOWLIST,
            ContractConstraint.LOG_ALL_OPERATIONS,
        },
        enabled=True,
        metadata={"category": "security", "phase": 1},
    )


def create_budget_tracking_contract() -> FeatureContract:
    """Create the standard budget tracking contract.
    
    This contract enforces resource budget management:
    - Track all resource consumption
    - Enforce budget limits
    - Log budget events
    """
    return FeatureContract(
        name="budget_tracking",
        version="1.0.0",
        description="Resource budget tracking and enforcement",
        required_tools={"budget_system"},
        optional_tools={"budget_callbacks"},
        constraints={
            ContractConstraint.ENFORCE_BUDGET_LIMITS,
            ContractConstraint.LOG_ALL_OPERATIONS,
        },
        enabled=True,
        metadata={"category": "resource_management", "phase": 3},
    )


def create_llm_calling_contract() -> FeatureContract:
    """Create the standard LLM calling contract.
    
    This contract enforces LLM API usage:
    - Track token usage
    - Enforce call limits
    - Log all LLM operations
    """
    return FeatureContract(
        name="llm_calling",
        version="1.0.0",
        description="LLM API calling with usage tracking",
        required_tools={"llm_client"},
        optional_tools={"llm_gemini", "llm_deepseek", "llm_ensemble"},
        constraints={
            ContractConstraint.TRACK_TOKEN_USAGE,
            ContractConstraint.ENFORCE_BUDGET_LIMITS,
            ContractConstraint.LOG_ALL_OPERATIONS,
        },
        enabled=True,
        metadata={"category": "llm", "phase": 3},
    )


def create_event_logging_contract() -> FeatureContract:
    """Create the standard event logging contract.
    
    This contract defines event system requirements:
    - All operations should be logged
    - Validate inputs before logging
    """
    return FeatureContract(
        name="event_logging",
        version="1.0.0",
        description="Structured event logging for observability",
        required_tools={"event_logger"},
        optional_tools={"event_persistence", "event_querying"},
        constraints={
            ContractConstraint.LOG_ALL_OPERATIONS,
            ContractConstraint.VALIDATE_INPUTS,
        },
        enabled=True,
        metadata={"category": "observability", "phase": 4},
    )


def register_standard_contracts(registry: ContractRegistry) -> None:
    """Register all standard contracts with a registry.
    
    Args:
        registry: The registry to register contracts with.
    """
    contracts = [
        create_shell_execution_contract(),
        create_budget_tracking_contract(),
        create_llm_calling_contract(),
        create_event_logging_contract(),
    ]
    
    for contract in contracts:
        registry.register(contract, replace=True)
    
    logger.info(f"Registered {len(contracts)} standard contracts")


# =============================================================================
# Global Registry
# =============================================================================

_global_registry: ContractRegistry | None = None
_global_validator: ContractValidator | None = None
_global_lock = threading.RLock()  # RLock allows reentrant acquisition


def get_global_registry() -> ContractRegistry:
    """Get or create the global contract registry."""
    global _global_registry
    with _global_lock:
        if _global_registry is None:
            _global_registry = ContractRegistry()
        return _global_registry


def set_global_registry(registry: ContractRegistry) -> None:
    """Set the global contract registry."""
    global _global_registry
    with _global_lock:
        _global_registry = registry


def get_global_validator() -> ContractValidator:
    """Get or create the global contract validator."""
    global _global_validator
    with _global_lock:
        if _global_validator is None:
            _global_validator = ContractValidator(get_global_registry())
        return _global_validator


def set_global_validator(validator: ContractValidator) -> None:
    """Set the global contract validator."""
    global _global_validator
    with _global_lock:
        _global_validator = validator


def validate_shell_execution_global(
    argv: list[str],
    shell: bool = False,
    operation: str = "shell_execution",
) -> None:
    """Validate shell execution using the global validator.
    
    This is a convenience function for modules that don't have
    direct access to the validator instance.
    """
    validator = get_global_validator()
    validator.validate_shell_execution(argv, shell, operation)


def reset_global_contracts() -> None:
    """Reset global contracts state (for testing)."""
    global _global_registry, _global_validator
    with _global_lock:
        _global_registry = None
        _global_validator = None
