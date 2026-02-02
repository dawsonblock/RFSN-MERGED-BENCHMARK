# RFSN Controller Security

This document describes the security architecture and guarantees of the RFSN Controller.

## Command Purity

**Guarantee**: All subprocess execution uses argv lists with `shell=False`. No shell string execution is permitted.

### Forbidden Patterns

The following patterns are **never** used in the RFSN codebase:

```python
# ❌ FORBIDDEN: shell=True
subprocess.run("echo hello", shell=True)

# ❌ FORBIDDEN: sh -c wrappers
subprocess.run(["sh", "-c", "echo hello"])

# ❌ FORBIDDEN: bash -c wrappers
subprocess.run(["bash", "-c", "echo hello"])

# ❌ FORBIDDEN: bash -lc wrappers
subprocess.run(["bash", "-lc", "echo hello"])
```

### Required Pattern

All commands must use explicit argv lists:

```python
# ✅ REQUIRED: argv list with shell=False
subprocess.run(["echo", "hello"], shell=False)
```

### Enforcement

1. **Static Analysis**: `tests/test_no_shell.py` scans all code for forbidden patterns
2. **Runtime Validation**: `exec_utils.safe_run()` rejects shell wrappers at runtime
3. **Code Review**: All PRs must pass the static analysis test

---

## Allowlist Enforcement

Commands are checked against allowlists **before** execution:

### Global Allowlist

- Defined in `command_allowlist.py`
- Applies to all host and container commands
- Blocks dangerous commands (rm -rf /, curl, wget, etc.)

### Language Allowlists

- Defined per buildpack in `allowlist_profiles.py`
- Only commands relevant to the language are permitted
- Example: Python projects allow `python`, `pip`, `pytest`

### Enforcement Order

```
Command Request
     │
     ▼
┌─────────────────┐
│ Global Allowlist│──▶ BLOCKED if dangerous
└─────────────────┘
     │ PASS
     ▼
┌─────────────────┐
│Language Allowlist│──▶ BLOCKED if not in list
└─────────────────┘
     │ PASS
     ▼
   EXECUTE
```

---

## Sandbox Isolation

### Docker Container

All code execution happens inside Docker containers:

- **Default**: No network access (`--network none`)
- **Read-only** root filesystem (except `/tmp`, `/repo`)
- **Resource limits**: CPU and memory capped
- **No privileged access**: `--security-opt=no-new-privileges`

### Environment Sanitization

Only safe environment variables are passed to subprocesses:

```python
SAFE_ENV_VARS = {"PATH", "HOME", "LANG", "PYTHONPATH", "TERM"}
```

All other variables (including API keys, credentials) are stripped.

### Path Jail

File operations are restricted to the sandbox directory:

- `pathlib.Path.resolve()` ensures no `../` traversal
- All paths validated before access
- Symlinks resolved and checked

---

## Audit Logging

All actions are logged to `events.jsonl`:

```json
{"timestamp": "2026-01-20T12:00:00Z", "type": "command_executed", "argv": ["pytest", "-q"], "exit_code": 0}
{"timestamp": "2026-01-20T12:00:01Z", "type": "patch_applied", "files": ["src/main.py"]}
```

Logs are:

- Append-only
- Structured JSON
- Include timestamps and full context

---

## Verification

Run the security test suite:

```bash
uv run pytest tests/test_no_shell.py -v
uv run pytest tests/test_zero_trust_hardening.py -v
```

All tests must pass before deployment.
