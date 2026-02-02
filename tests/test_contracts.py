"""Tests for the feature contracts system.

This module tests:
- FeatureContract dataclass creation and validation
- ContractViolation exception
- ContractRegistry registration and discovery
- ContractValidator enforcement
- Standard contracts
- Integration with controller and existing features
"""

import threading

import pytest

from rfsn_controller.contracts import (
    ContractConstraint,
    ContractRegistry,
    ContractValidator,
    ContractViolation,
    FeatureContract,
    create_budget_tracking_contract,
    create_event_logging_contract,
    create_llm_calling_contract,
    create_shell_execution_contract,
    get_global_registry,
    get_global_validator,
    register_standard_contracts,
    reset_global_contracts,
    set_global_registry,
    validate_shell_execution_global,
)

# =============================================================================
# FeatureContract Tests
# =============================================================================

class TestFeatureContract:
    """Tests for FeatureContract dataclass."""
    
    def test_create_basic_contract(self):
        """Test creating a basic feature contract."""
        contract = FeatureContract(
            name="test_feature",
            version="1.0.0",
            description="A test feature contract",
        )
        
        assert contract.name == "test_feature"
        assert contract.version == "1.0.0"
        assert contract.description == "A test feature contract"
        assert contract.enabled is True
        assert len(contract.required_tools) == 0
        assert len(contract.optional_tools) == 0
        assert len(contract.constraints) == 0
    
    def test_create_full_contract(self):
        """Test creating a contract with all fields."""
        contract = FeatureContract(
            name="full_feature",
            version="2.0.0",
            description="A complete feature contract",
            required_tools={"tool_a", "tool_b"},
            optional_tools={"tool_c"},
            constraints={
                ContractConstraint.NO_SHELL_TRUE,
                ContractConstraint.LOG_ALL_OPERATIONS,
            },
            enabled=True,
            metadata={"category": "security"},
        )
        
        assert contract.name == "full_feature"
        assert contract.version == "2.0.0"
        assert "tool_a" in contract.required_tools
        assert "tool_b" in contract.required_tools
        assert "tool_c" in contract.optional_tools
        assert ContractConstraint.NO_SHELL_TRUE in contract.constraints
        assert contract.metadata["category"] == "security"
    
    def test_contract_with_lists_converts_to_sets(self):
        """Test that list inputs are converted to sets."""
        contract = FeatureContract(
            name="list_test",
            version="1.0.0",
            description="Test list conversion",
            required_tools=["tool_a", "tool_b"],
            optional_tools=["tool_c"],
            constraints=[ContractConstraint.NO_SHELL_TRUE],
        )
        
        assert isinstance(contract.required_tools, set)
        assert isinstance(contract.optional_tools, set)
        assert isinstance(contract.constraints, set)
    
    def test_contract_validation_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            FeatureContract(
                name="",
                version="1.0.0",
                description="Invalid contract",
            )
    
    def test_contract_validation_empty_version(self):
        """Test that empty version raises ValueError."""
        with pytest.raises(ValueError, match="version cannot be empty"):
            FeatureContract(
                name="test",
                version="",
                description="Invalid contract",
            )
    
    def test_has_constraint(self):
        """Test has_constraint method."""
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
            constraints={ContractConstraint.NO_SHELL_TRUE},
        )
        
        assert contract.has_constraint(ContractConstraint.NO_SHELL_TRUE) is True
        assert contract.has_constraint(ContractConstraint.NO_SHELL_WRAPPERS) is False
    
    def test_requires_tool(self):
        """Test requires_tool method."""
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
            required_tools={"safe_run"},
            optional_tools={"docker_exec"},
        )
        
        assert contract.requires_tool("safe_run") is True
        assert contract.requires_tool("docker_exec") is False
        assert contract.requires_tool("unknown") is False
    
    def test_uses_tool(self):
        """Test uses_tool method."""
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
            required_tools={"safe_run"},
            optional_tools={"docker_exec"},
        )
        
        assert contract.uses_tool("safe_run") is True
        assert contract.uses_tool("docker_exec") is True
        assert contract.uses_tool("unknown") is False
    
    def test_to_dict(self):
        """Test to_dict serialization."""
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
            required_tools={"tool_a"},
            constraints={ContractConstraint.NO_SHELL_TRUE},
            metadata={"key": "value"},
        )
        
        data = contract.to_dict()
        
        assert data["name"] == "test"
        assert data["version"] == "1.0.0"
        assert "tool_a" in data["required_tools"]
        assert "no_shell_true" in data["constraints"]
        assert data["metadata"]["key"] == "value"
    
    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "description": "Test",
            "required_tools": ["tool_a"],
            "constraints": ["no_shell_true"],
            "enabled": True,
            "metadata": {"key": "value"},
        }
        
        contract = FeatureContract.from_dict(data)
        
        assert contract.name == "test"
        assert contract.version == "1.0.0"
        assert "tool_a" in contract.required_tools
        assert ContractConstraint.NO_SHELL_TRUE in contract.constraints


# =============================================================================
# ContractViolation Tests
# =============================================================================

class TestContractViolation:
    """Tests for ContractViolation exception."""
    
    def test_basic_violation(self):
        """Test creating a basic violation."""
        violation = ContractViolation(
            contract_name="shell_execution",
            constraint=ContractConstraint.NO_SHELL_TRUE,
            operation="subprocess_call",
        )
        
        assert violation.contract_name == "shell_execution"
        assert violation.constraint == "no_shell_true"
        assert violation.operation == "subprocess_call"
        assert "shell_execution" in str(violation)
        assert "no_shell_true" in str(violation)
    
    def test_violation_with_details(self):
        """Test violation with details and context."""
        violation = ContractViolation(
            contract_name="shell_execution",
            constraint="no_shell_true",
            operation="subprocess_call",
            details="shell=True was used",
            context={"command": ["bash", "-c", "echo test"]},
        )
        
        assert violation.details == "shell=True was used"
        assert violation.context["command"][0] == "bash"
        assert "shell=True was used" in str(violation)
    
    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = ContractViolation(
            contract_name="test",
            constraint=ContractConstraint.NO_SHELL_WRAPPERS,
            operation="exec",
            details="Details here",
            context={"key": "value"},
        )
        
        data = violation.to_dict()
        
        assert data["contract_name"] == "test"
        assert data["constraint"] == "no_shell_wrappers"
        assert data["operation"] == "exec"
        assert data["details"] == "Details here"
        assert data["context"]["key"] == "value"
        assert "timestamp" in data
    
    def test_violation_repr(self):
        """Test violation repr."""
        violation = ContractViolation(
            contract_name="test",
            constraint=ContractConstraint.NO_SHELL_TRUE,
            operation="exec",
        )
        
        repr_str = repr(violation)
        assert "ContractViolation" in repr_str
        assert "test" in repr_str


# =============================================================================
# ContractRegistry Tests
# =============================================================================

class TestContractRegistry:
    """Tests for ContractRegistry."""
    
    def test_register_contract(self):
        """Test registering a contract."""
        registry = ContractRegistry()
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
        )
        
        result = registry.register(contract)
        
        assert result is True
        assert registry.has_contract("test")
    
    def test_register_duplicate_returns_false(self):
        """Test registering duplicate contract returns False."""
        registry = ContractRegistry()
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
        )
        
        registry.register(contract)
        result = registry.register(contract)
        
        assert result is False
    
    def test_register_duplicate_with_replace(self):
        """Test registering duplicate with replace=True."""
        registry = ContractRegistry()
        contract1 = FeatureContract(
            name="test",
            version="1.0.0",
            description="Original",
        )
        contract2 = FeatureContract(
            name="test",
            version="2.0.0",
            description="Updated",
        )
        
        registry.register(contract1)
        result = registry.register(contract2, replace=True)
        
        assert result is True
        assert registry.get("test").version == "2.0.0"
    
    def test_unregister_contract(self):
        """Test unregistering a contract."""
        registry = ContractRegistry()
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
        )
        
        registry.register(contract)
        result = registry.unregister("test")
        
        assert result is True
        assert not registry.has_contract("test")
    
    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering nonexistent contract returns False."""
        registry = ContractRegistry()
        result = registry.unregister("nonexistent")
        
        assert result is False
    
    def test_get_contract(self):
        """Test getting a contract by name."""
        registry = ContractRegistry()
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
        )
        registry.register(contract)
        
        retrieved = registry.get("test")
        
        assert retrieved is not None
        assert retrieved.name == "test"
    
    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent contract returns None."""
        registry = ContractRegistry()
        result = registry.get("nonexistent")
        
        assert result is None
    
    def test_get_enabled(self):
        """Test getting only enabled contracts."""
        registry = ContractRegistry()
        contract1 = FeatureContract(
            name="enabled",
            version="1.0.0",
            description="Enabled",
            enabled=True,
        )
        contract2 = FeatureContract(
            name="disabled",
            version="1.0.0",
            description="Disabled",
            enabled=False,
        )
        
        registry.register(contract1)
        registry.register(contract2)
        
        enabled = registry.get_enabled()
        
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"
    
    def test_get_all(self):
        """Test getting all contracts."""
        registry = ContractRegistry()
        contract1 = FeatureContract(name="a", version="1.0.0", description="A")
        contract2 = FeatureContract(name="b", version="1.0.0", description="B")
        
        registry.register(contract1)
        registry.register(contract2)
        
        all_contracts = registry.get_all()
        
        assert len(all_contracts) == 2
    
    def test_get_by_constraint(self):
        """Test getting contracts by constraint."""
        registry = ContractRegistry()
        contract1 = FeatureContract(
            name="secure",
            version="1.0.0",
            description="Secure",
            constraints={ContractConstraint.NO_SHELL_TRUE},
        )
        contract2 = FeatureContract(
            name="insecure",
            version="1.0.0",
            description="Insecure",
        )
        
        registry.register(contract1)
        registry.register(contract2)
        
        contracts = registry.get_by_constraint(ContractConstraint.NO_SHELL_TRUE)
        
        assert len(contracts) == 1
        assert contracts[0].name == "secure"
    
    def test_get_by_tool(self):
        """Test getting contracts by tool."""
        registry = ContractRegistry()
        contract1 = FeatureContract(
            name="uses_safe_run",
            version="1.0.0",
            description="Uses safe_run",
            required_tools={"safe_run"},
        )
        contract2 = FeatureContract(
            name="no_tools",
            version="1.0.0",
            description="No tools",
        )
        
        registry.register(contract1)
        registry.register(contract2)
        
        contracts = registry.get_by_tool("safe_run")
        
        assert len(contracts) == 1
        assert contracts[0].name == "uses_safe_run"
    
    def test_enable_disable(self):
        """Test enabling and disabling contracts."""
        registry = ContractRegistry()
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
            enabled=True,
        )
        registry.register(contract)
        
        assert registry.is_enabled("test") is True
        
        registry.disable("test")
        assert registry.is_enabled("test") is False
        
        registry.enable("test")
        assert registry.is_enabled("test") is True
    
    def test_enable_nonexistent_returns_false(self):
        """Test enabling nonexistent contract returns False."""
        registry = ContractRegistry()
        result = registry.enable("nonexistent")
        
        assert result is False
    
    def test_check_dependencies(self):
        """Test checking contract dependencies."""
        registry = ContractRegistry()
        contract = FeatureContract(
            name="test",
            version="1.0.0",
            description="Test",
            required_tools={"tool_a", "tool_b", "tool_c"},
        )
        
        available = {"tool_a", "tool_b"}
        missing = registry.check_dependencies(contract, available)
        
        assert len(missing) == 1
        assert "tool_c" in missing
    
    def test_listener_notification(self):
        """Test that listeners are notified on registration."""
        registry = ContractRegistry()
        notifications: list[tuple] = []
        
        def listener(action: str, contract: FeatureContract) -> None:
            notifications.append((action, contract.name))
        
        registry.add_listener(listener)
        
        contract = FeatureContract(name="test", version="1.0.0", description="Test")
        registry.register(contract)
        
        assert len(notifications) == 1
        assert notifications[0] == ("registered", "test")
        
        registry.unregister("test")
        assert len(notifications) == 2
        assert notifications[1] == ("unregistered", "test")
    
    def test_clear_registry(self):
        """Test clearing all contracts."""
        registry = ContractRegistry()
        contract = FeatureContract(name="test", version="1.0.0", description="Test")
        registry.register(contract)
        
        registry.clear()
        
        assert len(registry.get_all()) == 0
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        registry = ContractRegistry()
        errors = []
        
        def register_contracts(prefix: str, count: int):
            try:
                for i in range(count):
                    contract = FeatureContract(
                        name=f"{prefix}_{i}",
                        version="1.0.0",
                        description="Test",
                    )
                    registry.register(contract)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=register_contracts, args=("a", 10)),
            threading.Thread(target=register_contracts, args=("b", 10)),
            threading.Thread(target=register_contracts, args=("c", 10)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(registry.get_all()) == 30


# =============================================================================
# ContractValidator Tests
# =============================================================================

class TestContractValidator:
    """Tests for ContractValidator."""
    
    def test_validate_shell_execution_allows_safe(self):
        """Test that safe shell execution is allowed."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        # Should not raise
        validator.validate_shell_execution(["ls", "-la"], shell=False)
    
    def test_validate_shell_execution_blocks_shell_true(self):
        """Test that shell=True is blocked."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        with pytest.raises(ContractViolation) as exc_info:
            validator.validate_shell_execution(["ls"], shell=True)
        
        assert "no_shell_true" in str(exc_info.value.constraint)
    
    def test_validate_shell_execution_blocks_sh_c(self):
        """Test that sh -c is blocked."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        with pytest.raises(ContractViolation) as exc_info:
            validator.validate_shell_execution(["sh", "-c", "echo test"])
        
        assert "no_shell_wrappers" in str(exc_info.value.constraint)
    
    def test_validate_shell_execution_blocks_bash_c(self):
        """Test that bash -c is blocked."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        with pytest.raises(ContractViolation) as exc_info:
            validator.validate_shell_execution(["bash", "-c", "echo test"])
        
        assert "no_shell_wrappers" in str(exc_info.value.constraint)
    
    def test_validate_shell_execution_blocks_interactive_shell(self):
        """Test that interactive shells are blocked."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        with pytest.raises(ContractViolation) as exc_info:
            validator.validate_shell_execution(["bash", "-i"])
        
        assert "no_interactive_shell" in str(exc_info.value.constraint)
    
    def test_validate_budget_operation_allows_within_limit(self):
        """Test budget operations within limit are allowed."""
        registry = ContractRegistry()
        registry.register(create_budget_tracking_contract())
        validator = ContractValidator(registry)
        
        # Should not raise
        validator.validate_budget_operation("llm_calls", current=5, limit=10)
    
    def test_validate_budget_operation_blocks_exceeded(self):
        """Test budget operations at limit are blocked."""
        registry = ContractRegistry()
        registry.register(create_budget_tracking_contract())
        validator = ContractValidator(registry)
        
        with pytest.raises(ContractViolation) as exc_info:
            validator.validate_budget_operation("llm_calls", current=10, limit=10)
        
        assert "enforce_budget_limits" in str(exc_info.value.constraint)
    
    def test_validate_operation_generic(self):
        """Test generic operation validation."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        # Shell operation should be validated
        with pytest.raises(ContractViolation):
            validator.validate_operation(
                "shell:subprocess",
                context={"argv": ["bash", "-c", "test"], "shell": False}
            )
    
    def test_is_operation_allowed(self):
        """Test is_operation_allowed check."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        assert validator.is_operation_allowed(
            "shell:test",
            context={"argv": ["ls"], "shell": False}
        ) is True
        
        assert validator.is_operation_allowed(
            "shell:test",
            context={"argv": ["bash", "-c", "test"], "shell": False}
        ) is False
    
    def test_violation_handler(self):
        """Test violation handler is called."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        violations = []
        
        def handler(v: ContractViolation):
            violations.append(v)
        
        validator.add_violation_handler(handler)
        
        try:
            validator.validate_shell_execution(["sh", "-c", "test"])
        except ContractViolation:
            pass
        
        assert len(violations) == 1
        assert violations[0].contract_name == "shell_execution"
    
    def test_remove_violation_handler(self):
        """Test removing violation handler."""
        registry = ContractRegistry()
        registry.register(create_shell_execution_contract())
        validator = ContractValidator(registry)
        
        violations = []
        
        def handler(v: ContractViolation):
            violations.append(v)
        
        validator.add_violation_handler(handler)
        validator.remove_violation_handler(handler)
        
        try:
            validator.validate_shell_execution(["sh", "-c", "test"])
        except ContractViolation:
            pass
        
        assert len(violations) == 0
    
    def test_disabled_contract_not_enforced(self):
        """Test that disabled contracts are not enforced."""
        registry = ContractRegistry()
        contract = create_shell_execution_contract()
        contract.enabled = False
        registry.register(contract)
        validator = ContractValidator(registry)
        
        # Should not raise even with shell=True because contract is disabled
        validator.validate_shell_execution(["ls"], shell=True)


# =============================================================================
# Standard Contracts Tests
# =============================================================================

class TestStandardContracts:
    """Tests for standard contract definitions."""
    
    def test_shell_execution_contract(self):
        """Test shell execution contract definition."""
        contract = create_shell_execution_contract()
        
        assert contract.name == "shell_execution"
        assert contract.version == "1.0.0"
        assert "safe_run" in contract.required_tools
        assert ContractConstraint.NO_SHELL_TRUE in contract.constraints
        assert ContractConstraint.NO_SHELL_WRAPPERS in contract.constraints
        assert ContractConstraint.NO_INTERACTIVE_SHELL in contract.constraints
    
    def test_budget_tracking_contract(self):
        """Test budget tracking contract definition."""
        contract = create_budget_tracking_contract()
        
        assert contract.name == "budget_tracking"
        assert contract.version == "1.0.0"
        assert "budget_system" in contract.required_tools
        assert ContractConstraint.ENFORCE_BUDGET_LIMITS in contract.constraints
    
    def test_llm_calling_contract(self):
        """Test LLM calling contract definition."""
        contract = create_llm_calling_contract()
        
        assert contract.name == "llm_calling"
        assert contract.version == "1.0.0"
        assert "llm_client" in contract.required_tools
        assert ContractConstraint.TRACK_TOKEN_USAGE in contract.constraints
    
    def test_event_logging_contract(self):
        """Test event logging contract definition."""
        contract = create_event_logging_contract()
        
        assert contract.name == "event_logging"
        assert contract.version == "1.0.0"
        assert "event_logger" in contract.required_tools
        assert ContractConstraint.LOG_ALL_OPERATIONS in contract.constraints
    
    def test_register_standard_contracts(self):
        """Test registering all standard contracts."""
        registry = ContractRegistry()
        register_standard_contracts(registry)
        
        assert registry.has_contract("shell_execution")
        assert registry.has_contract("budget_tracking")
        assert registry.has_contract("llm_calling")
        assert registry.has_contract("event_logging")


# =============================================================================
# Global Registry Tests
# =============================================================================

class TestGlobalRegistry:
    """Tests for global registry and validator."""
    
    def setup_method(self):
        """Reset global state before each test."""
        reset_global_contracts()
    
    def teardown_method(self):
        """Reset global state after each test."""
        reset_global_contracts()
    
    def test_get_global_registry(self):
        """Test getting global registry."""
        registry = get_global_registry()
        
        assert registry is not None
        assert isinstance(registry, ContractRegistry)
    
    def test_get_global_registry_singleton(self):
        """Test global registry is singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
    
    def test_set_global_registry(self):
        """Test setting global registry."""
        new_registry = ContractRegistry()
        contract = FeatureContract(name="test", version="1.0.0", description="Test")
        new_registry.register(contract)
        
        set_global_registry(new_registry)
        
        assert get_global_registry() is new_registry
        assert get_global_registry().has_contract("test")
    
    def test_get_global_validator(self):
        """Test getting global validator."""
        validator = get_global_validator()
        
        assert validator is not None
        assert isinstance(validator, ContractValidator)
    
    def test_validate_shell_execution_global(self):
        """Test global shell execution validation."""
        registry = get_global_registry()
        register_standard_contracts(registry)
        
        # Should not raise for safe command
        validate_shell_execution_global(["ls", "-la"])
        
        # Should raise for shell wrapper
        with pytest.raises(ContractViolation):
            validate_shell_execution_global(["sh", "-c", "test"])
    
    def test_reset_global_contracts(self):
        """Test resetting global contracts."""
        registry = get_global_registry()
        registry.register(
            FeatureContract(name="test", version="1.0.0", description="Test")
        )
        
        reset_global_contracts()
        
        # Should be a new registry without the test contract
        new_registry = get_global_registry()
        assert not new_registry.has_contract("test")


# =============================================================================
# Integration Tests
# =============================================================================

class TestContractIntegration:
    """Integration tests for contract system."""
    
    def setup_method(self):
        """Reset global state before each test."""
        reset_global_contracts()
    
    def teardown_method(self):
        """Reset global state after each test."""
        reset_global_contracts()
    
    def test_config_contract_config(self):
        """Test ContractsConfig in config module."""
        from rfsn_controller.config import ContractsConfig
        
        config = ContractsConfig()
        
        assert config.enabled is True
        assert config.shell_execution_enabled is True
        assert config.budget_tracking_enabled is True
        assert config.llm_calling_enabled is True
        assert config.event_logging_enabled is True
    
    def test_config_in_controller_config(self):
        """Test contracts field in ControllerConfig."""
        from rfsn_controller.config import ContractsConfig, ControllerConfig
        
        config = ControllerConfig(github_url="https://github.com/test/repo")
        
        assert hasattr(config, "contracts")
        assert isinstance(config.contracts, ContractsConfig)
    
    def test_context_has_contract_fields(self):
        """Test ControllerContext has contract registry and validator attributes."""
        from rfsn_controller.context import ControllerContext
        
        # Just check that the class has the expected attributes
        assert hasattr(ControllerContext, "contract_registry")
        assert hasattr(ControllerContext, "contract_validator")
    
    def test_contract_config_in_controller_config(self):
        """Test that ContractsConfig can be set in ControllerConfig."""
        from rfsn_controller.config import ContractsConfig, ControllerConfig
        
        contracts_config = ContractsConfig(
            enabled=True,
            shell_execution_enabled=False,
            budget_tracking_enabled=True,
        )
        config = ControllerConfig(
            github_url="https://github.com/test/repo",
            contracts=contracts_config,
        )
        
        assert config.contracts.enabled is True
        assert config.contracts.shell_execution_enabled is False
        assert config.contracts.budget_tracking_enabled is True
    
    def test_exec_utils_imports_contracts(self):
        """Test that exec_utils has contract integration code."""
        from pathlib import Path
        
        # Read exec_utils.py and verify it imports contracts
        exec_utils_path = Path(__file__).parent.parent / "rfsn_controller" / "exec_utils.py"
        content = exec_utils_path.read_text()
        
        # Should contain contract validation code
        assert "validate_shell_execution_global" in content
        assert "ContractViolation" in content
    
    def test_standalone_validator_workflow(self):
        """Test standalone validator workflow."""
        # Create registry and validator without context
        registry = ContractRegistry()
        register_standard_contracts(registry)
        validator = ContractValidator(registry)
        
        # Verify registry is initialized
        assert registry.has_contract("shell_execution")
        
        # Test validation works
        validator.validate_shell_execution(["ls", "-la"])
        
        with pytest.raises(ContractViolation):
            validator.validate_shell_execution(["bash", "-c", "test"])


# =============================================================================
# Contract Violation Handling Tests
# =============================================================================

class TestContractViolationHandling:
    """Tests for proper violation handling."""
    
    def setup_method(self):
        """Reset global state before each test."""
        reset_global_contracts()
    
    def teardown_method(self):
        """Reset global state after each test."""
        reset_global_contracts()
    
    def test_violation_includes_timestamp(self):
        """Test that violations include timestamp."""
        violation = ContractViolation(
            contract_name="test",
            constraint="test_constraint",
            operation="test_op",
        )
        
        assert violation.timestamp is not None
        assert "T" in violation.timestamp  # ISO format
    
    def test_violation_context_preserved(self):
        """Test that context is preserved in violations."""
        context = {"key1": "value1", "key2": ["a", "b", "c"]}
        
        violation = ContractViolation(
            contract_name="test",
            constraint="test_constraint",
            operation="test_op",
            context=context,
        )
        
        assert violation.context == context
    
    def test_multiple_contracts_can_enforce_same_constraint(self):
        """Test that multiple contracts can have the same constraint."""
        registry = ContractRegistry()
        
        contract1 = FeatureContract(
            name="contract1",
            version="1.0.0",
            description="First",
            constraints={ContractConstraint.NO_SHELL_TRUE},
        )
        contract2 = FeatureContract(
            name="contract2",
            version="1.0.0",
            description="Second",
            constraints={ContractConstraint.NO_SHELL_TRUE},
        )
        
        registry.register(contract1)
        registry.register(contract2)
        
        contracts = registry.get_by_constraint(ContractConstraint.NO_SHELL_TRUE)
        assert len(contracts) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
