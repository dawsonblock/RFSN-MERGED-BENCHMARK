# RFSN Controller Production Guide

## Quick Start

```bash
# Install
pip install -e .

# Run on a repository
rfsn-repair --repo https://github.com/user/repo --test "pytest"

# With planner v5
rfsn-repair --repo https://github.com/user/repo --test "pytest" --planner v5
```

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--repo` | Required | GitHub URL or local path |
| `--test` | `pytest -q` | Test command |
| `--planner` | `v5` | Planner version (v4, v5) |
| `--max-steps` | `50` | Maximum repair iterations |
| `--model` | `deepseek-r1` | LLM model |
| `--temps` | `0.0,0.2,0.4` | Temperature schedule |

### Environment Variables

```bash
# Required for LLM access
export DEEPSEEK_API_KEY=your_key
export GOOGLE_API_KEY=your_key  # For Gemini

# Optional
export RFSN_CACHE_DIR=~/.rfsn/cache
export RFSN_LOG_LEVEL=INFO
```

## Architecture

```
rfsn_controller/
├── controller.py       # Main orchestration
├── config.py           # Configuration dataclasses
├── buildpacks/         # Multi-language support
├── gates/              # Safety gates (PlanGate)
├── llm/                # LLM integrations
├── planner_v2/         # Hierarchical planner v4/v5
└── verification/       # Test verification system
```

## Key Features

- **Hierarchical Planning**: v4/v5 planners with goal decomposition
- **Multi-Model Support**: DeepSeek, Gemini, with fallback chains
- **Safety Gates**: PlanGate enforces invariants before execution
- **Outcome Learning**: Remembers what worked for similar bugs
- **Multi-Language**: Python, Node, Go, Rust, Java, C++, .NET

## Running Tests

```bash
# All tests
pytest tests/ -q

# Specific module
pytest tests/test_controller.py -v

# With coverage
pytest tests/ --cov=rfsn_controller
```

## Deployment

### Docker

```bash
docker build -t rfsn-controller .
docker run -e DEEPSEEK_API_KEY=xxx rfsn-controller --repo https://github.com/user/repo
```

### Production Checklist

- [ ] Set API keys in environment
- [ ] Configure log level
- [ ] Set cache directory
- [ ] Review security settings

## Troubleshooting

### Common Issues

1. **"SDK not available"**: Install missing SDK (`pip install openai` or `pip install google-genai`)
2. **Test failures**: Ensure test command is correct for the repository
3. **Timeout**: Increase `--max-steps` for complex bugs

### Logs

Logs are written to `~/.rfsn/logs/` by default.

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: See `docs/` directory
