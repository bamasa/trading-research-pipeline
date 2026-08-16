.PHONY: help install lint fmt typecheck test cov demo audit clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment and install the package with dev extras
	uv sync --all-extras

lint:  ## Check formatting and lint rules
	uv run ruff format --check .
	uv run ruff check .

fmt:  ## Apply formatting and auto-fixable lint rules
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Run static type checking
	uv run mypy

test:  ## Run the test suite
	uv run pytest

cov:  ## Run the test suite with a coverage report
	uv run pytest --cov --cov-report=term-missing

demo:  ## Generate the synthetic demo dataset and validate it
	uv run trading-research generate-demo-data --output data/demo
	uv run trading-research validate-data --input data/demo

audit:  ## Scan the working tree for private paths, secrets and heavy files
	uv run python scripts/audit_repo.py

clean:  ## Remove caches and generated data
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf data artifacts runs reports
