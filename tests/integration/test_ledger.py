from collections.abc import Iterator

import psycopg
import pytest

from demand_intel.config import settings
from demand_intel.pipelines.ingestion.ledger import ensure_ledger, libpq_url, record
from demand_intel.pipelines.ingestion.m5 import IngestionReport, ingest_m5
from tests.conftest import EXPECT, M5Dirs

pytestmark = pytest.mark.integration


def _sql(query: str, checksums: list[str]) -> int:
    with psycopg.connect(libpq_url(settings.database_url)) as conn:
        cur = conn.execute(query, (checksums,))
        if cur.description is None:
            return cur.rowcount
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture
def report(m5_dirs: M5Dirs) -> Iterator[IngestionReport]:
    """The fixture dataset's report; its ledger rows are removed so the real ledger stays clean."""
    report = ingest_m5(m5_dirs.raw, m5_dirs.out, m5_dirs.quarantine, expectations=EXPECT)
    checksums = [f.checksum_sha256 for f in report.files]
    delete = "DELETE FROM meta.ingestion_runs WHERE checksum_sha256 = ANY(%s)"
    ensure_ledger(settings.database_url)
    _sql(delete, checksums)
    yield report
    _sql(delete, checksums)


def test_recording_is_idempotent(report: IngestionReport) -> None:
    checksums = [f.checksum_sha256 for f in report.files]
    count = "SELECT count(*) FROM meta.ingestion_runs WHERE checksum_sha256 = ANY(%s)"

    assert record(settings.database_url, report) == len(report.files)
    assert record(settings.database_url, report) == 0
    assert _sql(count, checksums) == len(report.files)
