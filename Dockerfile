# RFSN Controller - portable Docker image (works on Apple Silicon)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy packaging first for better cache
COPY pyproject.toml README.md /app/

# Copy source (minimum required for install + tests)
COPY rfsn_controller/ /app/rfsn_controller/
COPY tests/ /app/tests/
COPY docs/ /app/docs/
COPY scripts/ /app/scripts/

# Install
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
 && python -m pip install --no-cache-dir -e ".[llm,dev]"

# Default entrypoint
ENTRYPOINT ["rfsn"]
