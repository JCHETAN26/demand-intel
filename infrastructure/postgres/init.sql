-- Separate database for MLflow's backend store; schemas for the warehouse layers.
CREATE DATABASE mlflow;

\connect demand_intel
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS forecasts;

-- Ingestion ledger (Phase 5): one row per ingested source file; checksum makes reruns idempotent.
CREATE TABLE IF NOT EXISTS meta.ingestion_runs (
    run_id          UUID PRIMARY KEY,
    dataset         TEXT        NOT NULL,
    source          TEXT        NOT NULL,
    source_file     TEXT        NOT NULL,
    checksum_sha256 TEXT        NOT NULL,
    schema_version  TEXT        NOT NULL,
    row_count       BIGINT      NOT NULL,
    rejected_count  BIGINT      NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset, source_file, checksum_sha256)
);
