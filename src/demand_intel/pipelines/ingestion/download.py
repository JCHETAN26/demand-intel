"""Download source datasets into `data/raw/<dataset>/` and record a checksum manifest.

Downloads are idempotent: if every expected file exists and matches the manifest, nothing
is fetched. Raw files are made read-only after download so they are never edited in place.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST = "_manifest.json"
CHUNK = 1 << 20

M5_COMPETITION = "m5-forecasting-accuracy"
M5_FILES = (
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def read_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    path = dataset_dir / MANIFEST
    if not path.exists():
        return None
    manifest: dict[str, Any] = json.loads(path.read_text())
    return manifest


def is_complete(dataset_dir: Path, filenames: Collection[str]) -> bool:
    """True if every expected file exists and matches the recorded checksum."""
    manifest = read_manifest(dataset_dir)
    if manifest is None:
        return False
    recorded = manifest.get("files", {})
    for name in filenames:
        path = dataset_dir / name
        if not path.exists() or name not in recorded:
            return False
        if recorded[name]["sha256"] != sha256_file(path):
            return False
    return True


def kaggle_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the Kaggle CLI.

    Newer Kaggle API tokens (prefix `KGAT`) are read from `KAGGLE_API_TOKEN`; accept one
    stored under the legacy `KAGGLE_KEY` name too.
    """
    env = dict(os.environ if environ is None else environ)
    key = env.get("KAGGLE_KEY", "")
    if not env.get("KAGGLE_API_TOKEN") and key.startswith("KGAT"):
        env["KAGGLE_API_TOKEN"] = key
    return env


def _kaggle_executable() -> str:
    local = Path(sys.executable).parent / "kaggle"
    if local.exists():
        return str(local)
    found = shutil.which("kaggle")
    if found is None:
        raise RuntimeError("Kaggle CLI not found; run `uv sync` to install it")
    return found


def _install(tmp_path: Path, dataset_dir: Path, filenames: Collection[str]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(filenames):
        target = dataset_dir / name
        if target.exists():
            target.chmod(stat.S_IWUSR | stat.S_IRUSR)
        (tmp_path / name).replace(target)
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        files[name] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    return files


def download_m5(raw_dir: Path, force: bool = False) -> Path:
    """Fetch the M5 competition files via the Kaggle CLI (requires accepted competition rules)."""
    dataset_dir = raw_dir / "m5"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_complete(dataset_dir, M5_FILES):
        return dataset_dir

    with tempfile.TemporaryDirectory(dir=dataset_dir) as tmp:
        tmp_path = Path(tmp)
        subprocess.run(
            [_kaggle_executable(), "competitions", "download", "-c", M5_COMPETITION, "-p", tmp],
            check=True,
            env=kaggle_env(),
        )
        payload = tmp_path / f"{M5_COMPETITION}.zip"
        payload_sha = sha256_file(payload)
        with zipfile.ZipFile(payload) as zf:
            members = {Path(n).name: n for n in zf.namelist() if Path(n).name in M5_FILES}
            missing = set(M5_FILES) - members.keys()
            if missing:
                raise RuntimeError(f"m5: archive is missing {sorted(missing)}")
            for name, member in members.items():
                with zf.open(member) as src, (tmp_path / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
        files = _install(tmp_path, dataset_dir, M5_FILES)

    manifest = {
        "dataset": "m5",
        "source": f"kaggle competition {M5_COMPETITION}",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "payload_sha256": payload_sha,
        "files": files,
    }
    (dataset_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset_dir
