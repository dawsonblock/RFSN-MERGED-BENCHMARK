# Feature Mode Reference

The RFSN controller operates in one of four feature modes, each optimized for different tasks.

## Available Modes

### `analysis` Mode

**Purpose**: Understand codebase without making changes.

**Actions**:

- Index repository structure
- Map dependencies
- Identify potential issues
- Generate reports

**Planner Behavior**: Read-only nodes, no patch generation.

---

### `repair` Mode (Default)

**Purpose**: Fix failing tests and bugs.

**Actions**:

- Run tests to identify failures
- Analyze error messages
- Generate minimal patches
- Verify fixes

**Planner Behavior**: Standard DAG with analyze → gather → patch → verify nodes.

**Policy Influence**: Arms like `search_stacktrace`, `apply_patch`, `rollback` are prioritized.

---

### `refactor` Mode

**Purpose**: Improve code structure without changing behavior.

**Actions**:

- Identify code smells
- Extract functions/classes
- Rename for clarity
- Update imports

**Planner Behavior**: Includes verification that tests still pass after each change.

**Policy Influence**: Arms like `refactor_module`, `fix_imports` are prioritized.

---

### `feature` Mode

**Purpose**: Add new functionality.

**Actions**:

- Parse feature requirements
- Design implementation approach
- Write new code
- Add tests

**Planner Behavior**: Extended DAG with understand → design → implement → verify nodes.

**Policy Influence**: Arms like `add_tests`, `read_file` are prioritized.

---

## Planner Integration

When `planner_mode=dag` is enabled, the planner generates a mode-specific DAG:

```
repair mode:
  analyze → gather_context → generate_patch → verify

feature mode:
  understand → design → implement → verify
```

Each node has:

- **Preconditions**: What must be true before execution
- **Actions**: Steps to perform (ordered by policy)
- **Verification**: How to confirm success

---

## Policy Integration

When `policy_mode=bandit` is enabled, the Thompson Sampling policy:

1. Orders actions within each planner node
2. Learns from outcomes to improve future ordering
3. Persists state to `learning_db_path`

Example: If `apply_patch` frequently succeeds after `search_stacktrace`, the policy will learn to prefer this sequence.

---

## CLI Usage

```bash
# Repair mode (default)
rfsn run --repo https://github.com/org/repo --test "pytest -q"

# Feature mode
rfsn run --repo https://... --feature-mode feature

# With planner
rfsn run --repo https://... --planner-mode dag

# Plan only (no execution)
rfsn plan --repo https://... --problem "Fix auth bug" --out plan.json

# Evaluate only
rfsn eval --repo https://... --test "pytest -q"
```
