.PHONY: install test lint format build clean

# Default python interpreter
PYTHON := python3

install:
	uv venv --python 3.11
	uv pip install -e ".[dev,llm]"

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy .

format:
	uv run ruff format .

build:
	docker build -t rfsn .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
