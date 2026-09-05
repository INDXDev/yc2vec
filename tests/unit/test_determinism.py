"""End-to-end determinism of the published build.

The release gate "deterministic build output except documented
timestamps/manifests" is only meaningful if something actually checks it. This
runs the whole fixture pipeline twice into separate directories and compares
the bytes.

It is the slowest unit test in the suite (a few seconds) and worth it: a
non-deterministic artifact means every dataset refresh produces a spurious diff,
which in turn means nobody reads the diffs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.versions import PUBLIC_ARTIFACT_VERSION

#: Fields that are expected to differ between runs, and why.
VOLATILE_MANIFEST_FIELDS = {
    "generated_at": "when this release was built",
    "source_retrieved_at": "when the source records were fetched",
}

ARTIFACTS = (
    "points.json",
    "companies.json",
    "tags.json",
    "taxonomy.json",
    "clusters.json",
    "search/docs.json",
)


def run_pipeline(data_dir: Path) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.cli",
            "run",
            "--profile",
            "fixture",
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return data_dir / "public" / f"v{PUBLIC_ARTIFACT_VERSION}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("determinism")
    return run_pipeline(base / "first"), run_pipeline(base / "second")


def test_published_artifacts_are_byte_identical(two_runs):
    first, second = two_runs
    for name in ARTIFACTS:
        assert digest(first / name) == digest(second / name), f"{name} is not deterministic"


def test_detail_shards_are_byte_identical(two_runs):
    first, second = two_runs
    for shard in sorted((first / "detail").glob("*.json")):
        other = second / "detail" / shard.name
        assert other.exists(), f"{shard.name} missing from the second run"
        assert digest(shard) == digest(other), f"detail/{shard.name} is not deterministic"


def test_manifest_differs_only_in_documented_timestamps(two_runs):
    first, second = two_runs
    a = json.loads((first / "manifest.json").read_text())
    b = json.loads((second / "manifest.json").read_text())

    differing = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    unexpected = differing - VOLATILE_MANIFEST_FIELDS.keys()
    assert not unexpected, (
        f"manifest fields changed between identical runs: {sorted(unexpected)}. "
        "Either the build is not deterministic, or the field belongs in "
        "VOLATILE_MANIFEST_FIELDS with a reason."
    )
    # The version identifiers in particular must be stable, since consumers key on them.
    for field in (
        "dataset_version",
        "embedding_space_version",
        "projection_version",
        "ontology_version",
    ):
        assert a[field] == b[field], f"{field} changed between identical runs"


def test_checksums_in_the_manifest_match_the_files(two_runs):
    first, _ = two_runs
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["checksums"], "the manifest published no checksums"
    for rel, recorded in manifest["checksums"].items():
        assert digest(first / rel)[:16] == recorded, f"{rel} checksum does not match its bytes"
