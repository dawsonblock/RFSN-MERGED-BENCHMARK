# RFSN Controller Configuration Guide

Complete reference for configuring RFSN Controller v0.2.0+

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Variables](#environment-variables)
3. [Configuration File](#configuration-file)
4. [LLM Configuration](#llm-configuration)
5. [Planner Configuration](#planner-configuration)
6. [Cache Configuration](#cache-configuration)
7. [Safety Configuration](#safety-configuration)
8. [Observability Configuration](#observability-configuration)
9. [CLI Options](#cli-options)
10. [Examples](#examples)

---

## Quick Start

### Using Environment Variables

```bash
# Set core configuration
export RFSN_LLM__PRIMARY="deepseek-chat"
export RFSN_PLANNER__MODE="v5"
export RFSN_CACHE__ENABLED="true"

# Run RFSN
rfsn --repo https://github.com/user/repo --test "pytest"
```

### Using Python API

```python
from rfsn_controller.config import RFSNConfig

config = RFSNConfig(
    llm__primary="deepseek-chat",
    planner__mode="v5",
    cache__enabled=True
)
```

---

## Environment Variables

All environment variables use the `RFSN_` prefix with `__` as a nested delimiter.

### Format

```
RFSN_<SECTION>__<KEY>=value
```

### Example

```bash
export RFSN_LLM__PRIMARY="deepseek-chat"
export RFSN_LLM__TEMPERATURE="0.2"
export RFSN_PLANNER__MAX_ITERATIONS="50"
```

---

## Configuration File

### Using .env File

Create `.env` in your project root:

```env
# LLM Configuration
RFSN_LLM__PRIMARY=deepseek-chat
RFSN_LLM__FALLBACK=gemini-2.0-flash
RFSN_LLM__TEMPERATURE=0.2
RFSN_LLM__MAX_TOKENS=4096

# Planner Configuration
RFSN_PLANNER__MODE=v5
RFSN_PLANNER__MAX_ITERATIONS=50
RFSN_PLANNER__RISK_BUDGET=3

# Cache Configuration
RFSN_CACHE__ENABLED=true
RFSN_CACHE__TTL_HOURS=72
RFSN_CACHE__MAX_SIZE_MB=1024

# Safety Configuration
RFSN_SAFETY__DOCKER_ENABLED=true
RFSN_SAFETY__GATE_STRICT_MODE=true

# Observability
RFSN_OBSERVABILITY__METRICS_ENABLED=true
RFSN_OBSERVABILITY__METRICS_PORT=9090
```

---

## LLM Configuration

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `primary` | string | `"deepseek-chat"` | Primary LLM model |
| `fallback` | string | `"gemini-2.0-flash"` | Fallback model |
| `temperature` | float | `0.2` | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | `4096` | Max tokens per response |
| `timeout` | float | `60.0` | API timeout in seconds |
| `max_retries` | int | `3` | Max retry attempts |

### Supported Models

**Primary Models:**
- `deepseek-chat` (recommended for planning)
- `gpt-4-turbo`
- `claude-3-opus`
- `gemini-2.0-pro`

**Fallback Models:**
- `gemini-2.0-flash` (fast, cost-effective)
- `gpt-3.5-turbo`
- `claude-3-haiku`

### Example

```bash
export RFSN_LLM__PRIMARY="gpt-4-turbo"
export RFSN_LLM__FALLBACK="gpt-3.5-turbo"
export RFSN_LLM__TEMPERATURE="0.1"
export RFSN_LLM__MAX_TOKENS="8192"
```

---

## Planner Configuration

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | `"v5"` | Planner version (`v4`, `v5`) |
| `max_plan_steps` | int | `12` | Max steps per plan |
| `max_iterations` | int | `50` | Max repair iterations |
| `max_stuck_iterations` | int | `3` | Iterations before stuck detection |
| `risk_budget` | int | `3` | Maximum risk level |

### Planner Modes

**v5 (Recommended):**
- SWE-bench optimized
- Structured proposal system
- Gate-first safety
- 50-60% solve rate

**v4 (Legacy):**
- Traditional planning
- Backward compatible

### Example

```bash
export RFSN_PLANNER__MODE="v5"
export RFSN_PLANNER__MAX_ITERATIONS="100"
export RFSN_PLANNER__RISK_BUDGET="5"
```

---

## Cache Configuration

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable caching |
| `ttl_hours` | int | `72` | Cache TTL in hours |
| `dir` | path | `~/.cache/rfsn` | Cache directory |
| `max_size_mb` | int | `1024` | Max cache size (MB) |
| `memory_enabled` | bool | `true` | In-memory tier |
| `disk_enabled` | bool | `true` | Disk tier |
| `semantic_enabled` | bool | `false` | Semantic tier |

### Cache Tiers

1. **Memory Cache** (fastest)
   - In-process caching
   - LRU eviction
   - No persistence

2. **Disk Cache** (persistent)
   - SQLite-based
   - Survives restarts
   - Configurable TTL

3. **Semantic Cache** (intelligent)
   - Vector similarity matching
   - Requires `sentence-transformers`
   - Best for similar but non-identical queries

### Example

```bash
export RFSN_CACHE__ENABLED="true"
export RFSN_CACHE__TTL_HOURS="168"  # 1 week
export RFSN_CACHE__MAX_SIZE_MB="2048"
export RFSN_CACHE__SEMANTIC_ENABLED="true"
```

**Install semantic cache:**
```bash
pip install 'rfsn-controller[semantic]'
```

---

## Safety Configuration

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `docker_enabled` | bool | `true` | Use Docker sandboxing |
| `max_risk_budget` | int | `3` | Max cumulative risk |
| `shell_allowed` | bool | `false` | Allow shell=True (⚠️ NOT RECOMMENDED) |
| `eval_allowed` | bool | `false` | Allow eval/exec (⚠️ NOT RECOMMENDED) |
| `gate_strict_mode` | bool | `true` | Strict gate validation |

### Security Recommendations

**✅ Recommended Settings:**
```bash
export RFSN_SAFETY__DOCKER_ENABLED="true"
export RFSN_SAFETY__GATE_STRICT_MODE="true"
export RFSN_SAFETY__SHELL_ALLOWED="false"
export RFSN_SAFETY__EVAL_ALLOWED="false"
```

**⚠️ Development Only (NOT for production):**
```bash
export RFSN_SAFETY__DOCKER_ENABLED="false"  # Local testing only
```

---

## Observability Configuration

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metrics_enabled` | bool | `true` | Enable Prometheus metrics |
| `metrics_port` | int | `9090` | Metrics endpoint port |
| `tracing_enabled` | bool | `false` | Enable distributed tracing |
| `jaeger_host` | string | `"localhost"` | Jaeger agent host |
| `jaeger_port` | int | `6831` | Jaeger agent port |
| `structured_logging` | bool | `true` | JSON logging |
| `log_level` | string | `"INFO"` | Log level |

### Prometheus Metrics

```bash
export RFSN_OBSERVABILITY__METRICS_ENABLED="true"
export RFSN_OBSERVABILITY__METRICS_PORT="9090"
```

Access metrics at: `http://localhost:9090/metrics`

### Distributed Tracing

```bash
# Install tracing dependencies
pip install 'rfsn-controller[observability]'

# Configure
export RFSN_OBSERVABILITY__TRACING_ENABLED="true"
export RFSN_OBSERVABILITY__JAEGER_HOST="localhost"
export RFSN_OBSERVABILITY__JAEGER_PORT="6831"
```

### Logging

```bash
export RFSN_OBSERVABILITY__LOG_LEVEL="DEBUG"
export RFSN_OBSERVABILITY__STRUCTURED_LOGGING="true"
```

**Log Levels:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

---

## CLI Options

### Basic Usage

```bash
rfsn --repo <repo_url> --test <test_command> [options]
```

### Available Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--repo` | `-r` | string | required | Repository URL or path |
| `--test` | `-t` | string | `"pytest"` | Test command |
| `--planner` | `-p` | string | `"v5"` | Planner version |
| `--llm` | | string | `"deepseek-chat"` | LLM model |
| `--max-iterations` | | int | `50` | Max iterations |
| `--docker/--no-docker` | | flag | `true` | Use Docker |
| `--cache/--no-cache` | | flag | `true` | Use caching |
| `--metrics/--no-metrics` | | flag | `true` | Enable metrics |
| `--verbose` | `-v` | flag | `false` | Verbose output |
| `--debug` | | flag | `false` | Debug mode |
| `--dry-run` | | flag | `false` | Dry run (no changes) |
| `--output` | `-o` | string | `"text"` | Output format (text/json/yaml) |

### Examples

**Basic repair:**
```bash
rfsn --repo https://github.com/user/repo --test "pytest tests/"
```

**With custom planner:**
```bash
rfsn --repo ./my-project --planner v5 --max-iterations 100
```

**Debug mode:**
```bash
rfsn --repo ./my-project --debug --verbose
```

**Dry run:**
```bash
rfsn --repo ./my-project --dry-run
```

**JSON output:**
```bash
rfsn --repo ./my-project --output json > result.json
```

---

## Examples

### 1. Production Configuration

```bash
# .env
RFSN_LLM__PRIMARY=deepseek-chat
RFSN_LLM__FALLBACK=gemini-2.0-flash
RFSN_PLANNER__MODE=v5
RFSN_PLANNER__MAX_ITERATIONS=50
RFSN_CACHE__ENABLED=true
RFSN_CACHE__TTL_HOURS=72
RFSN_SAFETY__DOCKER_ENABLED=true
RFSN_SAFETY__GATE_STRICT_MODE=true
RFSN_OBSERVABILITY__METRICS_ENABLED=true
RFSN_OBSERVABILITY__METRICS_PORT=9090
```

### 2. Development Configuration

```bash
# .env
RFSN_LLM__PRIMARY=gemini-2.0-flash  # Faster for dev
RFSN_PLANNER__MODE=v5
RFSN_CACHE__ENABLED=true
RFSN_SAFETY__DOCKER_ENABLED=false  # Local testing
RFSN_OBSERVABILITY__LOG_LEVEL=DEBUG
```

### 3. CI/CD Configuration

```bash
# GitHub Actions
env:
  RFSN_LLM__PRIMARY: ${{ secrets.LLM_MODEL }}
  RFSN_PLANNER__MODE: v5
  RFSN_PLANNER__MAX_ITERATIONS: 30
  RFSN_CACHE__ENABLED: true
  RFSN_SAFETY__DOCKER_ENABLED: true
  RFSN_OBSERVABILITY__METRICS_ENABLED: false
```

### 4. High-Performance Configuration

```bash
# .env
RFSN_LLM__PRIMARY=deepseek-chat
RFSN_LLM__MAX_TOKENS=8192
RFSN_PLANNER__MAX_ITERATIONS=100
RFSN_CACHE__ENABLED=true
RFSN_CACHE__SEMANTIC_ENABLED=true
RFSN_CACHE__MAX_SIZE_MB=4096
RFSN_OBSERVABILITY__METRICS_ENABLED=true
RFSN_OBSERVABILITY__TRACING_ENABLED=true
```

---

## Validation

### Check Configuration

```python
from rfsn_controller.config import get_config

config = get_config()
issues = config.validate_environment()

if issues:
    for issue in issues:
        print(f"Warning: {issue}")
else:
    print("✓ Configuration is valid")
```

### Verify Dependencies

```bash
# Check semantic cache
python -c "import sentence_transformers; print('✓ Semantic cache available')"

# Check observability
python -c "import opentelemetry; print('✓ Tracing available')"

# Check metrics
python -c "import prometheus_client; print('✓ Metrics available')"
```

---

## Troubleshooting

### Cache Not Working

1. Check cache directory exists and is writable
2. Verify TTL settings
3. Check disk space

```bash
ls -la ~/.cache/rfsn/
df -h ~/.cache/rfsn/
```

### Metrics Not Available

1. Verify port is not in use
2. Check firewall settings
3. Ensure prometheus-client installed

```bash
pip install 'rfsn-controller[observability]'
netstat -an | grep 9090
```

### Docker Issues

1. Verify Docker is running
2. Check Docker permissions
3. Test Docker access

```bash
docker ps
docker run hello-world
```

---

## Additional Resources

- [Main README](../README.md)
- [Planner v5 Documentation](../rfsn_controller/planner_v5/README.md)
- [Additional Upgrades Guide](../ADDITIONAL_UPGRADES.md)
- [GitHub Repository](https://github.com/dawsonblock/RFSN-GATE-CLEANED)

---

**Version**: 0.2.0  
**Last Updated**: January 29, 2026
