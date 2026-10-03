"""Ingestion ledger in Postgres: one row per ingested source file per distinct content.

The ledger records source, ingestion time, schema version, row counts, checksum and where
rejected rows went. Recording is idempotent: the same file content under the same schema
version is recorded once, however often ingestion is re-run.
"""

import psycopg

from demand_intel.pipelines.ingestion.m5 import IngestionReport

DDL = """
CREATE SCHEMA IF NOT EXISTS meta;
CREATE TABLE IF NOT EXISTS meta.ingestion_runs (
    run_id          uuid        PRIMARY KEY,
    dataset         text        NOT NULL,
    source_file     text        NOT NULL,
    checksum_sha256 text        NOT NULL,
    schema_version  text        NOT NULL,
    row_count       bigint      NOT NULL,
    loaded_count    bigint      NOT NULL,
    rejected_count  bigint      NOT NULL,
    rejected_path   text,
    outputs         text[]      NOT NULL,
    ingested_at     timestamptz NOT NULL,
    UNIQUE (dataset, source_file, checksum_sha256, schema_version)
);
"""


def libpq_url(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def record(database_url: str, report: IngestionReport) -> int:
    """Insert the report's files into the ledger. Returns the number of new rows."""
    inserted = 0
    with psycopg.connect(libpq_url(database_url)) as conn, conn.transaction():
        conn.execute(DDL)
        for f in report.files:
            cur = conn.execute(
                "INSERT INTO meta.ingestion_runs (run_id, dataset, source_file, checksum_sha256, "
                "schema_version, row_count, loaded_count, rejected_count, rejected_path, outputs, "
                "ingested_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (dataset, source_file, checksum_sha256, schema_version) DO NOTHING",
                (
                    f.run_id,
                    report.dataset,
                    f.source_file,
                    f.checksum_sha256,
                    f.schema_version,
                    f.row_count,
                    f.loaded_count,
                    f.rejected_count,
                    f.rejected_path,
                    f.outputs,
                    report.ingested_at,
                ),
            )
            inserted += cur.rowcount
    return inserted
