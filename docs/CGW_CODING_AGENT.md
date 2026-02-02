# CGW Coding Agent Architecture

> Serial decision controller for autonomous software engineering

## Overview

The CGW (Conscious Global Workspace) Coding Agent implements a **serial decision architecture** that enforces:

- **One decision per cycle** (single-slot workspace)
- **No parallel decisions** (thalamic gate arbitration)
- **Atomic commit only** (CGW runtime)
- **Blocking execution** (no tool overlap)

This is a control system, not a chatbot.

---

## Quick Start

### CLI Usage

```bash
# Run with goal
python -m cgw_ssl_guard.coding_agent.cli --goal "Fix failing tests"

# Run with config file
python -m cgw_ssl_guard.coding_agent.cli --config cgw.yaml

# Run with dashboard
python -m cgw_ssl_guard.coding_agent.cli --goal "Fix tests" --dashboard

# Generate default config
python -m cgw_ssl_guard.coding_agent.cli --init-config > cgw.yaml
```

### Python API

```python
from cgw_ssl_guard.coding_agent import IntegratedCGWAgent

# Simple usage
agent = IntegratedCGWAgent(goal="Fix failing tests")
result = agent.run()
print(result.summary())

# With config file
agent = IntegratedCGWAgent.from_config("cgw.yaml")
result = agent.run()
```

---

## Configuration

### Config File (cgw.yaml)

```yaml
agent:
  max_cycles: 100
  max_patches: 10
  goal: "Fix failing tests"

llm:
  provider: deepseek
  model: deepseek-coder
  temperature: 0.2

sandbox:
  image: "python:3.11-slim"
  timeout: 300

bandit:
  enabled: true
  exploration_bonus: 0.1

memory:
  enabled: true
  regression_threshold: 0.2

dashboard:
  enabled: true
  http_port: 8765
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CGW_LLM_PROVIDER` | LLM provider (deepseek/openai/gemini) |
| `CGW_MAX_CYCLES` | Maximum decision cycles |
| `CGW_GOAL` | Agent goal |
| `CGW_DASHBOARD_PORT` | Dashboard HTTP port |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CGW Coding Agent                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Bandit    │    │   Memory    │    │  Dashboard  │     │
│  │  (Strategy) │    │ (Outcomes)  │    │ (WebSocket) │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Integrated Runtime                      │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │            Coding Agent Runtime             │    │   │
│  │  │  ┌────────┐  ┌────────┐  ┌────────────┐    │    │   │
│  │  │  │ Propo- │→ │Thalamic│→ │    CGW     │    │    │   │
│  │  │  │  sals  │  │  Gate  │  │  Runtime   │    │    │   │
│  │  │  └────────┘  └────────┘  └─────┬──────┘    │    │   │
│  │  │                                │           │    │   │
│  │  │                          ┌─────▼──────┐    │    │   │
│  │  │                          │  Executor  │    │    │   │
│  │  │                          │ (Blocking) │    │    │   │
│  │  │                          └────────────┘    │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                      ┌─────▼─────┐                          │
│                      │Event Store│                          │
│                      │ (SQLite)  │                          │
│                      └───────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 2 Features

### Strategy Bandit

Adaptive action selection using UCB/Thompson Sampling:

```python
from cgw_ssl_guard.coding_agent import get_cgw_bandit, record_action_outcome

bandit = get_cgw_bandit()
boost = bandit.get_saliency_boost("RUN_TESTS")  # 0.8-1.5x
record_action_outcome("RUN_TESTS", success=True)
```

### Action Memory

Similarity-based proposal boosting with regression firewall:

```python
from cgw_ssl_guard.coding_agent import get_action_memory

memory = get_action_memory()
if memory.is_blocked("APPLY_PATCH", patch_hash):
    skip_action()  # Regression firewall blocks known-bad patterns
```

### Event Store

Persistent SQLite event logging:

```python
from cgw_ssl_guard.coding_agent import get_event_store

store = get_event_store()
store.start_session("session_123", goal="Fix tests")
store.export_session_json("session_123", "events.json")
```

### WebSocket Dashboard

Real-time monitoring with token cost tracking:

```python
from cgw_ssl_guard.coding_agent import get_dashboard

dashboard = get_dashboard()  # http://localhost:8765
dashboard.emit_event({"event_type": "CGW_COMMIT", "cycle_id": 1})
```

### Streaming LLM

Token-by-token generation with safety detection:

```python
from cgw_ssl_guard.coding_agent import SyncStreamingWrapper

for chunk in SyncStreamingWrapper().stream("Generate code"):
    print(chunk, end="")
```

---

## Action Types

| Action | Category | Description |
|--------|----------|-------------|
| `RUN_TESTS` | Execution | Run test suite |
| `ANALYZE_FAILURE` | Analysis | Parse test failures |
| `GENERATE_PATCH` | Analysis | Request LLM patch |
| `APPLY_PATCH` | Modification | Apply diff to codebase |
| `VALIDATE` | Validation | Run validation checks |
| `FINALIZE` | Terminal | End successfully |
| `ABORT` | Terminal | End with failure |

---

## One Full Cycle

```
1. COLLECT PROPOSALS
   Each generator analyzes context and submits Candidates
   Bandit provides saliency boosts

2. GATE SELECTION
   Gate scores candidates (saliency + urgency + surprise)
   Memory blocks regression-prone actions
   Forced signals bypass competition

3. CGW COMMIT
   Winner committed atomically to workspace
   Event emitted to store + dashboard

4. EXECUTE (BLOCKING)
   Executor runs action synchronously
   No other action can run until complete

5. UPDATE CONTEXT
   Results update proposal context
   Outcomes recorded to memory + bandit

6. → NEXT CYCLE
```

---

## Testing

```bash
# All tests
pytest tests/cgw/ -v

# E2E tests (18)
pytest tests/cgw/test_e2e.py -v

# Phase 2 tests (37)
pytest tests/cgw/test_phase2.py -v

# Total: 55 tests
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `coding_agent_runtime.py` | Core serial decision loop |
| `integrated_runtime.py` | Full-featured runtime with all integrations |
| `config.py` | YAML/JSON configuration |
| `cli.py` | Command-line interface |
| `cgw_bandit.py` | Strategy learning |
| `action_memory.py` | Outcome memory + regression firewall |
| `event_store.py` | SQLite event persistence |
| `streaming_llm.py` | Async LLM streaming |
| `websocket_dashboard.py` | Real-time dashboard |
| `llm_adapter.py` | Multi-provider LLM client |
| `docker_sandbox.py` | Container execution |
| `cgw_metrics.py` | Prometheus metrics |
