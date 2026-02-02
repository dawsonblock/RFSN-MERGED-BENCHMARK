# Docker sandbox mode (Mac M2 + API)

This repo can execute verification steps in a container using the built-in `docker_run()` helpers.

## Why you want this
- Untrusted repos/test commands run **inside Docker**, not on macOS.
- You keep policy guardrails (allowlist + contracts) **and** add OS-level isolation.

## Build the image
From repo root:

```bash
docker build -t rfsn-coding:latest .
```

## Quick smoke test

```bash
docker run --rm rfsn-coding:latest rfsn --help
```

## Running verification in Docker (example)
The controller uses `rfsn_controller.sandbox.docker_test()` and `docker_run()`.

Typical pattern:
- Set your API key (DeepSeek uses OpenAI-compatible client in this repo)
- Run the controller; it will run verify/test commands via docker when configured

```bash
export DEEPSEEK_API_KEY=...
# then run your normal rfsn invocation (see README)
```

## Security defaults
- Prefer `read_only=True` when mounting repos.
- Prefer `network=False` unless you explicitly need package downloads.
- Avoid shell commands; pass argv (space-separated strings get parsed by `shlex.split`).

