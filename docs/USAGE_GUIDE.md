# RFSN Controller Usage Guide

This guide covers the features introduced in Stage 1 and Stage 2: shell scanner utility, budget configuration, secure subprocess execution, event logging, and feature contracts.

## Table of Contents

### Stage 1
1. [Shell Scanner Utility](#shell-scanner-utility)
2. [Budget Configuration](#budget-configuration)
3. [Secure Subprocess Execution](#secure-subprocess-execution)
4. [Testing Your Code](#testing-your-code)

### Stage 2
5. [Event Logging System](#event-logging-system)
6. [Feature Contracts](#feature-contracts)
7. [Querying Events](#querying-events)
8. [Contract Enforcement](#contract-enforcement)

---

## Shell Scanner Utility

The shell scanner is a static analysis tool that detects potentially unsafe shell execution patterns in Python code.

### Command-Line Interface

#### Basic Usage
```bash
# Scan a directory
python -m rfsn_controller.shell_scanner /path/to/code

# Scan specific files
python -m rfsn_controller.shell_scanner file1.py file2.py

# Scan current directory
python -m rfsn_controller.shell_scanner .
```

#### Output Formats
```bash
# Text output (default)
python -m rfsn_controller.shell_scanner src/

# JSON output for programmatic processing
python -m rfsn_controller.shell_scanner src/ --json

# GitHub Actions annotations
python -m rfsn_controller.shell_scanner src/ --format github-actions
```

#### CI/CD Mode
```bash
# Exit with code 1 if violations found (for CI pipelines)
python -m rfsn_controller.shell_scanner src/ --ci
```

#### Exclusions
```bash
# Exclude specific directories
python -m rfsn_controller.shell_scanner src/ --exclude-dir tests --exclude-dir vendor

# Exclude specific files
python -m rfsn_controller.shell_scanner src/ --exclude-file legacy_code.py

# Exclude by regex pattern
python -m rfsn_controller.shell_scanner src/ --exclude-pattern ".*_test\.py"
```

### What It Detects

| Pattern | Severity | Example |
|---------|----------|---------|
| `shell=True` | CRITICAL | `subprocess.run(cmd, shell=True)` |
| `os.system()` | CRITICAL | `os.system("rm -rf /")` |
| `os.popen()` | HIGH | `os.popen("ls")` |
| Interactive shells | CRITICAL | `subprocess.run(["bash", "-i"])` |
| Shell wrappers | CRITICAL | `subprocess.run(["sh", "-c", cmd])` |

### Programmatic Usage

```python
from rfsn_controller.shell_scanner import ShellScanner, ScanResult

# Create scanner
scanner = ShellScanner()

# Scan a file
result: ScanResult = scanner.scan_file("/path/to/file.py")
print(f"Violations: {len(result.violations)}")

# Scan a directory
result = scanner.scan_directory("/path/to/project", exclude_dirs=["tests"])

# Check results
if not result.clean:
    for v in result.violations:
        print(f"{v.file}:{v.line}: [{v.severity}] {v.message}")
```

---

## Budget Configuration

Budget gates prevent resource exhaustion by limiting steps, LLM calls, tokens, time, and subprocess calls.

### CLI Configuration

```bash
# Set maximum steps
python -m rfsn_controller.cli --max-steps 100

# Set LLM limits
python -m rfsn_controller.cli --max-llm-calls 50 --max-tokens 100000

# Set time limit (seconds)
python -m rfsn_controller.cli --max-time 3600

# Set subprocess limit
python -m rfsn_controller.cli --max-subprocess-calls 200

# Set warning threshold (default 0.8 = 80%)
python -m rfsn_controller.cli --budget-warning-threshold 0.75

# Combined example
python -m rfsn_controller.cli \
    --max-steps 50 \
    --max-llm-calls 20 \
    --max-tokens 50000 \
    --max-time 1800 \
    --max-subprocess-calls 100
```

### Programmatic Configuration

```python
from rfsn_controller.config import BudgetConfig, ControllerConfig

# Create budget configuration
budget = BudgetConfig(
    max_steps=100,
    max_llm_calls=50,
    max_tokens=100000,
    max_time_seconds=3600,
    max_subprocess_calls=200,
    warning_threshold=0.8
)

# Use in controller config
config = ControllerConfig(
    budget=budget,
    # ... other config options
)
```

### Direct Budget Usage

```python
from rfsn_controller.budget import Budget, BudgetState, BudgetExceeded

# Create a budget
budget = Budget(
    max_steps=10,
    max_llm_calls=5,
    max_tokens=10000,
    max_subprocess_calls=20,
    warning_threshold=0.8
)

# Record usage
try:
    budget.record_step()
    budget.record_llm_call(tokens=1500)
    budget.record_subprocess_call()
except BudgetExceeded as e:
    print(f"Budget exceeded: {e.resource} - {e.message}")

# Check state
state = budget.get_state()
if state == BudgetState.WARNING:
    print("Warning: Approaching budget limits")
elif state == BudgetState.EXCEEDED:
    print("Budget exceeded!")

# Get usage summary
summary = budget.get_usage_summary()
print(summary)
# Output:
# {'steps': {'used': 1, 'max': 10, 'remaining': 9},
#  'llm_calls': {'used': 1, 'max': 5, 'remaining': 4},
#  'tokens': {'used': 1500, 'max': 10000, 'remaining': 8500},
#  'subprocess_calls': {'used': 1, 'max': 20, 'remaining': 19},
#  'state': 'active',
#  'elapsed_seconds': 0.001}
```

### Callbacks for Warnings and Exceedances

```python
from rfsn_controller.budget import Budget

def on_warning(resource: str, used: int, limit: int):
    print(f"⚠️ Warning: {resource} at {used}/{limit}")

def on_exceeded(resource: str, used: int, limit: int):
    print(f"🛑 Exceeded: {resource} at {used}/{limit}")

budget = Budget(
    max_steps=10,
    warning_threshold=0.8,
    on_warning_callback=on_warning,
    on_exceeded_callback=on_exceeded
)
```

### Global Budget (for modules without context)

```python
from rfsn_controller.budget import (
    set_global_budget,
    get_global_budget,
    record_subprocess_call_global,
    record_llm_call_global
)

# Set global budget (done automatically by context.create_context)
budget = Budget(max_subprocess_calls=100)
set_global_budget(budget)

# Use global functions from any module
record_subprocess_call_global()  # Tracks subprocess call
record_llm_call_global(tokens=500)  # Tracks LLM call
```

---

## Secure Subprocess Execution

Stage 1 introduces secure subprocess execution patterns that prevent shell injection.

### Using exec_utils

```python
from rfsn_controller.exec_utils import safe_run, CommandResult

# Run a command safely
result: CommandResult = safe_run(["ls", "-la", "/tmp"])
print(f"Exit code: {result.returncode}")
print(f"Output: {result.stdout}")

# With timeout
result = safe_run(["sleep", "5"], timeout=2.0)

# With custom environment
result = safe_run(
    ["python", "script.py"],
    env={"PYTHONPATH": "/custom/path"}
)
```

### Using SubprocessPool

```python
from rfsn_controller.optimizations import get_subprocess_pool, CommandResult

pool = get_subprocess_pool()

# Run command through pool (manages concurrency)
result: CommandResult = pool.run_command(
    ["git", "status"],
    timeout=30,
    cwd="/path/to/repo"
)

if result.success:
    print(result.stdout)
else:
    print(f"Error: {result.stderr}")
```

### What NOT to Do

```python
# ❌ WRONG - shell=True allows injection
subprocess.run("ls " + user_input, shell=True)

# ❌ WRONG - shell wrappers are rejected
safe_run(["sh", "-c", "ls -la"])

# ❌ WRONG - interactive shells are rejected
safe_run(["/bin/bash", "-i"])

# ✅ CORRECT - use argument lists
safe_run(["ls", "-la", user_input])
```

---

## Testing Your Code

### Running All Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rfsn_controller --cov-report=term-missing

# Run only security tests
pytest tests/ -m security

# Run only unit tests
pytest tests/ -m unit

# Run shell scanner tests
pytest tests/ -m scanner
```

### Testing Budget Integration

```python
import pytest
from rfsn_controller.budget import Budget, BudgetExceeded, set_global_budget

class TestMyFeature:
    def test_respects_budget_limits(self):
        """Ensure feature respects budget limits."""
        budget = Budget(max_subprocess_calls=2)
        set_global_budget(budget)
        
        # Your code that makes subprocess calls
        # ...
        
        # Third call should raise
        with pytest.raises(BudgetExceeded):
            # call_that_makes_subprocess()
            pass
```

### Testing Shell Security

```python
import pytest
from rfsn_controller.optimizations import get_subprocess_pool

class TestMySubprocessUsage:
    def test_rejects_shell_wrappers(self):
        """Ensure shell wrappers are rejected."""
        pool = get_subprocess_pool()
        
        with pytest.raises(ValueError, match="shell wrapper"):
            pool.run_command(["sh", "-c", "echo hello"])
    
    def test_accepts_safe_commands(self):
        """Ensure safe commands work."""
        pool = get_subprocess_pool()
        result = pool.run_command(["echo", "hello"])
        assert result.success
```

---

## Best Practices

1. **Always use argument lists** instead of string commands
2. **Set appropriate budget limits** for your use case
3. **Monitor budget state** and handle warnings gracefully
4. **Run the shell scanner** in CI pipelines with `--ci` flag
5. **Write security tests** for any code that executes subprocesses
6. **Use `safe_run()`** instead of raw `subprocess.run()`

---

## Event Logging System

Stage 2 introduces a comprehensive event logging system for agent observability.

### Basic Event Logging

```python
from rfsn_controller.events import (
    EventLogger, EventType, EventSeverity,
    set_global_logger, get_global_logger
)

# Create a logger
logger = EventLogger(run_id="my-run-001", max_events=10000)

# Set as global logger (for use across modules)
set_global_logger(logger)

# Log events
event = logger.log(
    event_type=EventType.CONTROLLER_STEP,
    source="my_module",
    data={"step_number": 1, "action": "initialize"},
    severity=EventSeverity.INFO
)

print(f"Event ID: {event.event_id}")
```

### Convenience Logging Methods

```python
from rfsn_controller.events import EventLogger

logger = EventLogger()

# Log controller step
logger.log_controller_step(step_number=1, phase="planning")

# Log LLM call
logger.log_llm_call(
    model="gpt-4",
    tokens_prompt=500,
    tokens_completion=1000,
    latency_ms=250.5,
    success=True
)

# Log budget warning
logger.log_budget_warning(
    resource="tokens",
    current=8000,
    limit=10000,
    percentage=0.8
)

# Log subprocess execution
logger.log_subprocess_exec(
    command=["git", "status"],
    success=True,
    exit_code=0,
    duration_ms=50.0
)

# Log security violation
logger.log_security_violation(
    violation_type="shell_injection",
    details="Attempted shell=True",
    context={"command": "ls -la"}
)

# Log error
logger.log_error(
    error_type="RuntimeError",
    message="Connection failed",
    traceback="..."
)
```

### Global Event Functions

```python
from rfsn_controller.events import (
    set_global_logger, get_global_logger, clear_global_logger,
    log_event_global, log_controller_step_global, log_llm_call_global,
    log_budget_warning_global, log_subprocess_exec_global,
    log_security_violation_global, log_error_global
)

# Set up global logger
logger = EventLogger(run_id="run-123")
set_global_logger(logger)

# Log from anywhere in the codebase
log_controller_step_global(step_number=5, phase="execution")
log_llm_call_global("claude-3", 100, 500, 150.0, True)
log_subprocess_exec_global(["echo", "hello"], True, 0, 10.0)
```

### Event Callbacks

```python
from rfsn_controller.events import EventLogger, EventType

def on_security_event(event):
    if event.event_type == EventType.SECURITY_VIOLATION:
        print(f"ALERT: {event.data}")
        # Send to monitoring system
        
logger = EventLogger()
logger.register_callback(on_security_event)

# Callback fires when security events are logged
logger.log_security_violation("shell_true", "Blocked shell=True")
```

---

## Feature Contracts

Contracts define and enforce agent capability requirements.

### Defining a Contract

```python
from rfsn_controller.contracts import (
    FeatureContract, ContractConstraint, ContractRegistry
)

# Define a contract
my_contract = FeatureContract(
    name="secure_executor",
    version="1.0.0",
    description="Secure command execution feature",
    required_tools={"exec_utils", "sandbox"},
    optional_tools={"docker"},
    constraints={
        ContractConstraint.NO_SHELL_TRUE,
        ContractConstraint.NO_SHELL_WRAPPERS,
        ContractConstraint.ENFORCE_BUDGET_LIMITS,
    },
    enabled=True,
    metadata={"author": "security-team"}
)

# Register with the registry
registry = ContractRegistry()
registry.register(my_contract)
```

### Using the Global Registry

```python
from rfsn_controller.contracts import (
    get_global_registry, set_global_registry,
    get_global_validator, set_global_validator,
    register_standard_contracts
)

# Get/create global registry
registry = get_global_registry()

# Register standard contracts (shell, budget, LLM, events)
register_standard_contracts(registry)

# Query contracts
shell_contracts = registry.get_by_constraint(ContractConstraint.NO_SHELL_TRUE)
print(f"Found {len(shell_contracts)} shell contracts")

# Check if contract exists
if registry.has_contract("shell_execution"):
    contract = registry.get("shell_execution")
    print(f"Shell contract v{contract.version}")
```

### Contract Discovery

```python
from rfsn_controller.contracts import ContractRegistry, ContractConstraint

registry = ContractRegistry()
# ... register contracts ...

# Get all enabled contracts
enabled = registry.get_enabled()

# Get contracts by constraint
budget_contracts = registry.get_by_constraint(ContractConstraint.ENFORCE_BUDGET_LIMITS)

# Get contracts that use a specific tool
exec_contracts = registry.get_by_tool("exec_utils")

# Check dependencies
contract = registry.get("my_feature")
missing = registry.check_dependencies(contract)
if missing:
    print(f"Missing dependencies: {missing}")
```

---

## Querying Events

The EventQuery class provides powerful filtering capabilities.

### Basic Queries

```python
from rfsn_controller.events import EventLogger, EventQuery, EventType, EventSeverity

logger = EventLogger()
# ... log some events ...

# Create a query
query = EventQuery(logger)

# Get all events
all_events = query.execute()

# Filter by event type
llm_events = query.by_type(EventType.LLM_CALL).execute()

# Filter by multiple types
important = query.by_types([
    EventType.SECURITY_VIOLATION,
    EventType.BUDGET_EXCEEDED,
    EventType.ERROR
]).execute()

# Filter by severity
errors = query.by_severity(EventSeverity.ERROR).execute()
warnings_and_above = query.by_min_severity(EventSeverity.WARNING).execute()
```

### Advanced Queries

```python
from datetime import datetime, timedelta, timezone
from rfsn_controller.events import EventQuery

# Time-based filtering
now = datetime.now(timezone.utc)
one_hour_ago = now - timedelta(hours=1)

recent_events = query.by_time_range(
    start=one_hour_ago.isoformat(),
    end=now.isoformat()
).execute()

# Filter by source
controller_events = query.by_source("controller").execute()

# Filter by data fields
high_token_calls = query.by_data_filter(
    "tokens_total", 
    lambda t: t > 5000
).execute()

# Combine filters
critical_recent = (
    query
    .by_min_severity(EventSeverity.ERROR)
    .by_time_range(start=one_hour_ago.isoformat())
    .by_types([EventType.SECURITY_VIOLATION, EventType.ERROR])
    .limit(100)
    .execute()
)
```

### Event Persistence with EventStore

```python
from rfsn_controller.events import EventStore, EventLogger
from pathlib import Path

# Create store
store = EventStore(Path("/var/log/rfsn/events.jsonl"))

# Append events from logger
logger = EventLogger()
# ... log events ...
store.append_batch(logger.get_events())

# Read events back
events = store.read_all()

# Iterate efficiently
for event in store.iter_events():
    print(f"{event.timestamp}: {event.event_type.value}")

# Rotate logs
store.rotate(max_files=5)

# Query from store
query = EventQuery.from_store(store)
violations = query.by_type(EventType.SECURITY_VIOLATION).execute()
```

---

## Contract Enforcement

The ContractValidator enforces constraints at runtime.

### Basic Validation

```python
from rfsn_controller.contracts import (
    ContractValidator, ContractRegistry, ContractViolation,
    register_standard_contracts
)

# Setup
registry = ContractRegistry()
register_standard_contracts(registry)
validator = ContractValidator(registry)

# Validate shell execution
try:
    # This passes - safe command
    validator.validate_shell_execution(["ls", "-la"])
    
    # This raises ContractViolation - shell wrapper
    validator.validate_shell_execution(["sh", "-c", "ls"])
except ContractViolation as e:
    print(f"Violation: {e.contract_name} - {e.constraint}")
    print(f"Details: {e.details}")
```

### Checking Without Exceptions

```python
from rfsn_controller.contracts import ContractValidator

validator = ContractValidator(registry)

# Check if operation is allowed
if validator.is_operation_allowed("shell_execution", {"argv": ["ls", "-la"]}):
    # Safe to proceed
    subprocess.run(["ls", "-la"])
else:
    print("Operation not allowed by contracts")
```

### Violation Handlers

```python
from rfsn_controller.contracts import ContractValidator, ContractViolation

def log_violation(violation: ContractViolation):
    print(f"VIOLATION: {violation.contract_name}")
    # Log to monitoring system
    # Send alert
    # etc.

validator = ContractValidator(registry)
validator.add_violation_handler(log_violation)

# Handler is called when violations occur
try:
    validator.validate_shell_execution(["bash", "-c", "cmd"])
except ContractViolation:
    pass  # Handler already processed it
```

### Integration with exec_utils

```python
from rfsn_controller.exec_utils import safe_run
from rfsn_controller.contracts import (
    get_global_validator, set_global_validator,
    ContractValidator, ContractRegistry, register_standard_contracts
)

# Setup global validator
registry = ContractRegistry()
register_standard_contracts(registry)
validator = ContractValidator(registry)
set_global_validator(validator)

# safe_run() automatically validates commands
try:
    # This works
    result = safe_run(["git", "status"])
    
    # This raises ContractViolation
    result = safe_run(["sh", "-c", "git status"])
except ContractViolation as e:
    print(f"Blocked: {e}")
```

### CLI Configuration

```bash
# Enable/disable specific contracts
python -m rfsn_controller.cli \
    --enable-shell-contract \
    --enable-budget-contract \
    --enable-llm-contract \
    --enable-event-contract

# Strict mode (all violations are fatal)
python -m rfsn_controller.cli --strict-contracts

# Disable all contracts (not recommended)
python -m rfsn_controller.cli --no-contracts
```

---

## Best Practices

### Stage 1
1. **Always use argument lists** instead of string commands
2. **Set appropriate budget limits** for your use case
3. **Monitor budget state** and handle warnings gracefully
4. **Run the shell scanner** in CI pipelines with `--ci` flag
5. **Write security tests** for any code that executes subprocesses
6. **Use `safe_run()`** instead of raw `subprocess.run()`

### Stage 2
7. **Enable event logging** for production observability
8. **Register contracts** for all critical features
9. **Set up violation handlers** to catch security issues
10. **Persist events** for audit trails and debugging
11. **Query events** to analyze agent behavior patterns
12. **Use callbacks** for real-time monitoring integration

---

*For more details, see [STAGE1_SUMMARY.md](STAGE1_SUMMARY.md), [STAGE2_SUMMARY.md](STAGE2_SUMMARY.md), and [MIGRATION_NOTES.md](MIGRATION_NOTES.md)*
