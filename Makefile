.PHONY: install lint format typecheck test check up down evaluate report

# On a filesystem without symlinks (e.g. exFAT), export UV_PROJECT_ENVIRONMENT=~/.venvs/demand-intel
# in your shell; otherwise uv uses ./.venv.
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

evaluate:       ## Regenerate every artifact in PROJECT_SPEC.md §10 from raw data (hours on 4 cores)
	uv sync --group dev --extra ml --extra deep --extra nlp --extra api
	uv run demand-intel ingest m5
	uv run demand-intel ingest weather
	uv run demand-intel ingest text
	uv run demand-intel splits
	uv run demand-intel eda
	scripts/run_forecasting.sh baselines statistical lgbm_full lgbm_panel ablation deep weather_oracle
	uv run demand-intel nlp-prepare
	uv run demand-intel nlp-train --models majority,rules,tfidf,embeddings,encoder
	uv run demand-intel nlp-signals --event-source encoder
	scripts/run_forecasting.sh multimodal
	$(MAKE) report
	uv run demand-intel promote --challenger lgbm --population full
	uv run demand-intel inventory --model lgbm
	uv run demand-intel forecast-batch
	uv run demand-intel monitor
	uv run demand-intel nlp-benchmark
	uv run python scripts/benchmark_api.py
	$(MAKE) report

report:         ## Rebuild tables, bundle extras and the acceptance report from existing runs (minutes)
	uv run demand-intel report
	uv run demand-intel bundle-extras
	uv run demand-intel acceptance
