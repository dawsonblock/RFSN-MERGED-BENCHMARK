# RFSN Benchmark

<div align="center">

[![Safety Kernel](https://img.shields.io/badge/Safety-RFSN%20Gate-green)](/)
[![SWE-bench](https://img.shields.io/badge/Benchmark-SWE--bench-blue)](https://swe-bench.github.io/)
[![Audit](https://img.shields.io/badge/Audit-Cryptographic%20Trace-purple)](/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The Gold Standard for Deterministic, Auditable Autonomous Code Repair.**

[Features](#key-features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Dashboard](#dashboard) • [Audit](#full-episode-audit)

</div>

---

## 🚀 Overview

**RFSN Benchmark** is a next-generation agent architecture designed for **SWE-bench** evaluation. It treats safety and determinism as first-class citizens, ensuring that every agent action is sandboxed, gated, and cryptographically recorded.

Unlike traditional agents that operate as "black boxes," RFSN provides a **Full Episode Audit** system that guarantees transparency. Every decision, proposed patch, and test result is hashed and chained, creating an immutable ledger of the repair process.

## ✨ Key Features

- **🛡️ Deterministic Safety Kernel**: The `PlanGate` enforces strict non-negotiable invariants (no shell injection, path restrictions) that no LLM can bypass.
- **📜 Full Episode Audit**: A cryptographic ledger records every PlanGate decision, applied patch hash, and sanitized test output. Replay any episode to verify its integrity bit-for-bit.
- **🧠 Upstream Intelligence**: Decoupled architecture supporting **Gemini**, **DeepSeek**, and **Ensemble** strategies. Intelligence modules (planning, search, learning) suggest repairs but never touch the kernel directly.
- **📊 Operational Dashboard**: A built-in Streamlit dashboard for real-time monitoring of agent runs, bandit learning rates, and system health.
- **⚡ Performance Optimized**: Features speculative execution, multi-tier caching, and a highly optimized episode runner loop.

## 🏗️ Architecture

RFSN enforces a **single authority path**. All intelligence flows through a strict gateway before touching the environment.

```mermaid
graph TD
    UI[Upstream Intelligence] -->|Proposes| GA[Gate Adapter]
    GA -->|Validates| PG[PlanGate (Kernel)]
    PG -->|Executes| ER[Episode Runner]
    ER -->|Records| AL[Audit Ledger]
    
    subgraph "Safe Zone"
    PG
    ER
    AL
    end
    
    subgraph "Intelligence"
    UI
    end
```

## 🏁 Quick Start

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/dawsonblock/RFSN-MERGED-BENCHMARK.git
cd RFSN-MERGED-BENCHMARK

# Install dependencies
pip install -e ".[dev]"
```

### Running a Benchmark Episode

Run the unified evaluation loop with a dataset task:

```bash
# Run a single task from SWE-bench Lite
python run_episode.py --task-id django__django-11001 --dataset datasets/swebench_lite.jsonl
```

### 🖥️ Dashboard

Monitor your agents in real-time:

```bash
streamlit run rfsn_dashboard/app.py
```

Access the dashboard at `http://localhost:8501`.

## 🔍 Full Episode Audit

RFSN introduces a **Trace-based Audit System**. Every run generates a `.trace` file containing a hash chain of events.

### Recording a Trace

Set `RFSN_TRACE_MODE=RECORD` to generate a trace file during execution.

### Verifying a Trace

To prove that a run was not tampered with, use the verification tool:

```bash
# Verify a specific trace file
python verify_audit.py --trace-file path/to/episode.trace
```

This ensures:

1. **Gate Decisions** match the recorded policy.
2. **Patches** match the exact hash of the applied code.
3. **Test Results** are reproducible and match the recorded output hash.

## ⚙️ Configuration

Control the agent behavior with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RFSN_STRICT_BENCH` | Enforce strict dataset validation | `1` |
| `RFSN_TRACE_MODE` | Audit mode: `RECORD`, `REPLAY`, or `OFF` | `OFF` |
| `DEEPSEEK_API_KEY` | Key for DeepSeek models | — |
| `GEMINI_API_KEY` | Key for Google Gemini models | — |
| `RFSN_LOG_LEVEL` | Logging verbosity | `INFO` |

## 🤝 Contributing

We welcome contributions! Please follow the **Architecture Invariants**:

1. **Never modify PlanGate directly.** Safety logic is immutable.
2. **Add tests** for all new upstream modules.
3. **Verify determinism** before submitting PRs.

## 📄 License

MIT © [Dawson Block](https://github.com/dawsonblock)
