import psycopg
import pytest

from demand_intel.config import settings
from demand_intel.pipelines.ingestion.ledger import libpq_url, record
from demand_intel.pipelines.ingestion.m5 import ingest_m5
from tests.conftest import EXPECT, M5Dirs

pytestmark = pytest.mark.integration


def _count(checksums: list[str]) -> int:
    with psycopg.connect(libpq_url(settings.database_url)) as conn:
        row = conn.execute(
            "SELECT count(*) FROM meta.ingestion_runs WHERE checksum_sha256 = ANY(%s)",
            (checksums,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_recording_is_idempotent(m5_dirs: M5Dirs) -> None:
    report = ingest_m5(m5_dirs.raw, m5_dirs.out, m5_dirs.quarantine, expectations=EXPECT)
    checksums = [f.checksum_sha256 for f in report.files]

    first = record(settings.database_url, report)
    again = record(settings.database_url, report)
    assert again == 0
    assert _count(checksums) == len(report.files)
    assert first in (0, len(report.files))  # 0 if a previous test run recorded them
