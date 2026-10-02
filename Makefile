.PHONY: install lint format typecheck test check up down evaluate

# The repo lives on an exFAT drive, which has no symlinks, so the virtualenv lives in $HOME.
# Override with `make install UV_PROJECT_ENVIRONMENT=.venv` on a normal filesystem.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/demand-intel
export UV_LINK_MODE ?= copy

install:        ## Core + dev deps
	uv sync --group dev
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not integration and not data and not gpu"

check: lint typecheck test

up:             ## Postgres + MLflow
	docker compose up -d --build

down:
	docker compose down

evaluate:       ## Regenerates every artifact in PROJECT_SPEC.md §10 (built up phase by phase)
	@echo "Not implemented yet — see PROJECT_SPEC.md §10" && exit 1
