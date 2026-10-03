-- Separate database for MLflow's backend store; schemas for the warehouse layers.
CREATE DATABASE mlflow;

\connect demand_intel
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS forecasts;

-- The ingestion ledger table (meta.ingestion_runs) is created by
-- demand_intel.pipelines.ingestion.ledger so its DDL has a single source.
