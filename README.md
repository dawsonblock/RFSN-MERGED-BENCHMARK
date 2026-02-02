# RFSN Benchmark

[![Safety Kernel](https://img.shields.io/badge/Safety-RFSN%20Gate-green)](/)
[![SWE-bench](https://img.shields.io/badge/Benchmark-SWE--bench-blue)](https://swe-bench.github.io/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

> **Safety-first autonomous code repair with upstream learning.**

RFSN Benchmark is a complete agent architecture for SWE-bench-class autonomous code repair. It combines a **deterministic safety kernel** (PlanGate) with **upstream intelligence modules** (planner, search, learning, retrieval) that never touch the gate.

## ⚠️ Important: Unified Architecture

This repository uses a **single authority path**:

```
Dataset → Episode Runner → Gate Adapter → PlanGate (kernel)
              ↑
         Upstream Intelligence (propose_v2)
              ↑
    ┌─────────┴─────────────┐
    │                       │
Repair Classification   Skill Routing
(taxonomy.py)          (router.py)
    │                       │
Failure Retrieval      Planner Selection
(failure_index.py)     (thompson.py)
```

**There is only ONE gate**: `rfsn_controller/gates/plan_gate.py`  
**There is only ONE eval path**: `eval/run_v2.py` → `orchestrator/episode_runner.py`

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UPSTREAM INTELLIGENCE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   Planner    │  │    Search    │  │   Learning   │  │  Retrieval  │  │
│  │  (planning)  │  │    (beam)    │  │  (Thompson)  │  │  (memory)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  │
│         │                 │                 │                 │         │
│         └─────────────────┴────────┬────────┴─────────────────┘         │
│                                    │                                    │
│                          ┌─────────▼─────────┐                          │
│                          │   PROPOSE V2      │                          │
│                          │ (agent/propose_v2)│                          │
│                          └─────────┬─────────┘                          │
│                                    │                                    │
│                          ┌─────────▼─────────┐                          │
│                          │ EPISODE RUNNER    │                          │
│                          │ (orchestrator/)   │                          │
│                          └─────────┬─────────┘                          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼  PROPOSALS ONLY
┌─────────────────────────────────────────────────────────────────────────┐
│                        DETERMINISTIC KERNEL                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ GATE ADAPTER │──│   PLAN GATE  │──│ Self-Critique│                   │
│  │(single route)│  │ (the kernel) │  │   (rubric)   │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                         │
│  ✓ Command allowlist    ✓ Path restrictions    ✓ No shell injection    │
│  ✓ Deterministic        ✓ Fail-closed         ✓ Append-only logging    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Critical: SWE-bench Correctness

This implementation follows the **correct SWE-bench procedure**:

1. **Clone repo** at `base_commit`
2. **Apply `test_patch`** (adds failing tests) - **MUST succeed or task is INVALID**
3. **Run tests** (baseline - should fail)
4. **Generate patches** via upstream intelligence
5. **Gate each proposal** through PlanGate
6. **Apply and test** - accept first passing result
7. **Reset between attempts** to clean state

```python
# The correct flow is in orchestrator/episode_runner.py
from orchestrator import run_one_task

result = run_one_task(
    task={"instance_id": "...", "repo": "...", "base_commit": "...", "test_patch": "..."},
    repo_url="https://github.com/owner/repo.git",
    llm_patch_fn=your_llm_function,
)
```

## Datasets (STRICT MODE)

**Strict mode is ON by default.** Missing datasets cause hard failure.

```bash
# Required: place ONE of these in datasets/
datasets/swebench_lite.jsonl
datasets/swebench_verified.jsonl
datasets/swebench_full.jsonl

# Copy a dataset in
python scripts/fetch_swebench.py /path/to/swebench_lite.jsonl

# Verify datasets are present
python scripts/verify_datasets.py
```

Set `RFSN_STRICT_BENCH=0` for local development without datasets.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run eval (requires dataset + LLM API keys)
python -m eval.run_v2 --dataset swebench_lite.jsonl --max-tasks 5
```

## Modules

### eval/ - Evaluation Infrastructure

- `strictness.py` - Strict mode enforcement
- `dataset_loader.py` - SWE-bench dataset loading
- `repo_setup.py` - Repository cloning and patch application
- `test_cmd.py` - Test command derivation
- `run_v2.py` - Unified eval entry point

### agent/ - Proposal Pipeline

- `gate_adapter.py` - Single gate routing (wraps PlanGate)
- `propose_v2.py` - Upstream intelligence integration

### orchestrator/ - Episode Execution

- `episode_runner.py` - Single authority loop
- `loop_v2.py` - Legacy loop (being deprecated)

### planner/ - Repair Planning

- `spec.py` - Plan and RepairStep types
- `planner.py` - Multi-step plan generation

### search/ - Patch Exploration

- `beam.py` - Beam search implementation
- `patch_search.py` - Search strategies

### learning/ - Bandit Selection

- `thompson.py` - Thompson sampling
- `planner_bandit.py` - Multi-planner selection
- `outcomes.py` - Outcome scoring

### repair/ - Bug Classification

- `taxonomy.py` - 16-category repair ontology
- `classifier.py` - Failure classification

### skills/ - Repo-Specific Routing

- `heads.py` - Skill head definitions
- `router.py` - Skill selection

### retrieval/ - Memory

- `failure_index.py` - Persistent failure patterns
- `recall.py` - Context building for prompts

### rfsn_controller/ - The Kernel

- `gates/plan_gate.py` - **THE gate** (single source of truth)
- `gates/self_critique.py` - 22-check rubric
- `exec_utils.py` - Hardened execution

## Your LLM Patch Function

You must implement `llm_patch_fn(plan, context)`:

```python
def your_llm_patch_fn(plan, context) -> list[dict]:
    """
    Generate patch candidates from a plan.
    
    Args:
        plan: Plan object with steps, metadata
        context: Dict with hypotheses, skill_heads, retrieval
        
    Returns:
        List of dicts, each with:
            - patch_text: Unified diff
            - summary: Short description
    """
    # Your DeepSeek/Gemini/Claude call here
    response = call_llm(
        plan=plan,
        hypotheses=context["hypotheses"],
        skills=context["skill_heads"],
        similar_fixes=context["retrieval"],
    )
    return [{"patch_text": response.patch, "summary": response.summary}]
```

## Safety Invariants

The kernel enforces these **non-negotiable** invariants:

1. **Serial Authority** — One proposal at a time
2. **Immutable Gating** — Gate cannot be bypassed
3. **Deterministic Validation** — Same input → same decision
4. **Fail-Closed** — Any error → reject
5. **Command Allowlist** — Only approved operations
6. **Path Restrictions** — No access outside workspace
7. **Append-Only Logging** — Full audit trail

## CI/CD

```yaml
# .github/workflows/bench.yml runs:
1. Verify datasets exist
2. Run benchmark with learning
3. Cache .rfsn_state for cross-run improvement
4. Upload results as artifacts
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RFSN_STRICT_BENCH` | Strict mode (fail on missing datasets) | `1` |
| `RFSN_LOG_LEVEL` | Logging verbosity | `INFO` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |
| `GEMINI_API_KEY` | Gemini API key | — |

## Tests

```bash
pytest tests/ -v
pytest tests/test_self_critique.py -v
```

## What's NOT This Repo

This repo is **not**:

- A sample task fallback (disabled in strict mode)
- A loose policy filter (everything goes through PlanGate)
- Two parallel stacks (unified to one authority path)

## Contributing

1. **Never modify PlanGate** — All intelligence is upstream
2. **Route through GateAdapter** — No duplicate gate logic
3. **Use episode_runner** — Single authority loop
4. **Add tests** — New modules need tests

## License

MIT
