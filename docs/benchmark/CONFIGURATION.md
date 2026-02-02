# Benchmark Configuration Guide

This document explains how to configure RFSN Benchmark for different evaluation scenarios.

## Configuration Options

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RFSN_BENCH_STRICT` | bool | `0` | Enable strict mode (required for official benchmarks) |
| `RFSN_MAX_WORKERS` | int | `4` | Maximum parallel workers |
| `RFSN_TASK_TIMEOUT` | int | `600` | Per-task timeout in seconds |
| `RFSN_OUTPUT_DIR` | path | `./runs` | Directory for results and artifacts |

### Strict Mode

When `RFSN_BENCH_STRICT=1`:

1. **No sample fallback** — Missing datasets cause fatal exit
2. **No error recovery** — Corrupted data stops the benchmark
3. **Machine-readable errors** — All errors have status codes
4. **Deterministic execution** — Same inputs produce same outputs

### Dataset Configuration

```bash
# SWE-bench Lite (300 tasks, faster)
python -m eval.cli --dataset swebench_lite

# SWE-bench Verified (500 tasks, gold standard)
python -m eval.cli --dataset swebench_verified

# Custom dataset
python -m eval.cli --dataset /path/to/tasks.jsonl
```

### Parallelization

```bash
# Serial execution (default)
python -m eval.cli --parallel 1

# 4 parallel workers with isolated worktrees
python -m eval.cli --parallel 4

# Maximum parallelization (uses all CPU cores)
python -m eval.cli --parallel auto
```

## CI/CD Integration

### GitHub Actions

```yaml
name: RFSN Benchmark
on:
  workflow_dispatch:
    inputs:
      dataset:
        description: 'Dataset to evaluate'
        default: 'swebench_lite'
      max_tasks:
        description: 'Maximum tasks'
        default: '10'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    env:
      RFSN_BENCH_STRICT: 1
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e '.[llm]'
      - run: python -m eval.cli --dataset ${{ inputs.dataset }} --max-tasks ${{ inputs.max_tasks }}
      - uses: actions/upload-artifact@v4
        with:
          name: results
          path: runs/
```

### Output Format

Results are saved to `runs/<run_id>/`:

```
runs/swebench_lite_1706745600/
├── results.json          # Aggregated results
├── summary.md            # Human-readable summary
├── tasks/
│   ├── django__django-11234/
│   │   ├── patch.diff    # Applied patch
│   │   ├── log.txt       # Execution log
│   │   └── evidence.json # Verification evidence
│   └── ...
└── artifacts/
    └── ...
```

## Troubleshooting

### Common Issues

**Dataset not found:**

```bash
# Ensure dataset exists
ls -la datasets/

# Download SWE-bench
python -m eval.swebench download
```

**Task timeout:**

```bash
# Increase timeout
export RFSN_TASK_TIMEOUT=1200
```

**Memory issues with parallel workers:**

```bash
# Reduce parallelism
python -m eval.cli --parallel 2
```
