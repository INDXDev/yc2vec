"""Gates that run against the published bundle alone.

These matter because the deploy job has only what is committed. A gate that
needs the intermediate tables cannot run there, so the shipping check has to be
expressible from the artifacts themselves -- which also means a consumer of the
dataset can run it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.models import (
    CompanyTagFeature,
    CompanyTagJudgment,
    EvidenceSpan,
    Neighbor,
    ReleaseManifest,
    Tag,
    UmapPoint,
)
from pipeline.normalize.companies import normalize_companies
from pipeline.publish.browser import publish_browser_artifacts
from pipeline.quality.published import run_published_gates
from pipeline.versions import PUBLIC_ARTIFACT_VERSION

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def gate(results, fragment):
    return next(r for r in results if fragment in r.name)


@pytest.fixture
def release(tmp_path, sample_raws) -> Path:
    companies = normalize_companies(sample_raws)[:10]
    ids = [c.company_id for c in companies]
    tags = [
        Tag(
            tag_id="alpha",
            canonical_name="Alpha",
            definition="A definition long enough to use.",
            facet="industry",
            state="active",
            ontology_version="1",
            created_at=NOW,
            updated_at=NOW,
        )
    ]
    judgment = CompanyTagJudgment(
        judgment_id="j",
        company_id=ids[0],
        tag_id="alpha",
        decision="yes",
        confidence=0.9,
        rationale="because",
        evidence=[EvidenceSpan(document_id="d", quote="q")],
        shortlist_reason="retrieval",
        model="m",
        prompt_version="p",
        ontology_version="1",
        pipeline_version="1",
        run_id="r",
        created_at=NOW,
    )
    feature = CompanyTagFeature(
        company_id=ids[0],
        tag_id="alpha",
        present=1,
        raw_confidence=0.9,
        calibrated_confidence=0.8,
        information_weight=0.5,
        feature_value=0.4,
        judgment_id="j",
        ontology_version="1",
        run_id="r",
    )
    manifest = ReleaseManifest(
        dataset_version="test",
        schema_version="1",
        public_artifact_version=PUBLIC_ARTIFACT_VERSION,
        pipeline_version="1",
        ontology_version="1",
        embedding_space_version="v1",
        projection_version="p1",
        generated_at=NOW,
        source_url="https://example.com",
        models={"chat": "m"},
        limitations=["tags are inferred"],
    )
    publish_browser_artifacts(
        tmp_path,
        companies=companies,
        tags=tags,
        features=[feature],
        judgments=[judgment],
        neighbors=[
            Neighbor(
                company_id=ids[0],
                neighbor_company_id=ids[1],
                space="combined",
                rank=0,
                similarity=0.8,
                embedding_space_version="v1",
            )
        ],
        points=[
            UmapPoint(
                company_id=c.company_id,
                x=float(i),
                y=float(-i),
                cluster_id=i % 2,
                projection_version="p1",
                embedding_space_version="v1",
            )
            for i, c in enumerate(companies)
        ],
        clusters=[],
        terms=[],
        mappings=[],
        manifest=manifest,
        quality={"companies": 10, "active_tags": 1},
    )
    return tmp_path


def root(release: Path) -> Path:
    return release / f"v{PUBLIC_ARTIFACT_VERSION}"


def rewrite(path: Path, mutate) -> None:
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def test_a_clean_release_passes_every_gate(release):
    assert all(r.passed for r in run_published_gates(release))


def test_missing_release_is_reported_not_crashed(tmp_path):
    results = run_published_gates(tmp_path)
    assert len(results) == 1 and not results[0].passed


def test_a_tampered_file_fails_its_checksum(release):
    (root(release) / "points.json").write_text('{"count": 0, "ids": []}')
    assert not gate(run_published_gates(release), "manifest checksum").passed


def test_a_point_without_a_company_row_fails(release):
    rewrite(root(release) / "companies.json", lambda d: d["rows"].pop())
    assert not gate(run_published_gates(release), "agree on who exists").passed


def test_mismatched_coordinate_arrays_fail(release):
    """A short array would silently mis-plot every point after the gap."""
    rewrite(root(release) / "points.json", lambda d: d["x"].pop())
    assert not gate(run_published_gates(release), "parallel and finite").passed


def test_a_null_coordinate_fails(release):
    """JSON has no Infinity, and orjson writes it as null rather than refusing.

    So a non-finite coordinate does not blow up at write time; it reaches the
    browser as a null that breaks the plot silently. The gate has to catch it.
    """
    import orjson

    assert orjson.dumps({"x": [float("inf")]}) == b'{"x":[null]}'  # documents the behaviour
    rewrite(root(release) / "points.json", lambda d: d["x"].__setitem__(0, None))
    assert not gate(run_published_gates(release), "parallel and finite").passed


def test_a_corrupted_artifact_fails_cleanly_rather_than_crashing(release):
    """A gate that raises reports a broken checker, not a broken release."""
    (root(release) / "points.json").write_bytes(b'{"x": [Infinity]}')
    results = run_published_gates(release)
    assert any(not r.passed for r in results)
    # And it must still be a list of gate results, not an exception.
    assert all(hasattr(r, "passed") for r in results)


def test_a_reference_to_an_unpublished_tag_fails(release):
    rewrite(
        root(release) / "companies.json",
        lambda d: d["rows"][0].update({"T": ["ghost"], "S": [0.5]}),
    )
    assert not gate(run_published_gates(release), "referenced tag is published").passed


def test_misaligned_tag_scores_fail(release):
    rewrite(
        root(release) / "companies.json", lambda d: d["rows"][0].update({"T": ["alpha"], "S": []})
    )
    assert not gate(run_published_gates(release), "same length").passed


def test_a_neighbour_pointing_outside_the_release_fails(release):
    def mutate(d):
        for record in d.values():
            for entries in (record.get("neighbors") or {}).values():
                for entry in entries:
                    entry["id"] = "ycoss:does-not-exist"

    # Only one company in the fixture has neighbours, and which shard it lands
    # in is a hash; rewriting every shard avoids depending on that.
    for shard in (root(release) / "detail").glob("*.json"):
        rewrite(shard, mutate)
    assert not gate(run_published_gates(release), "published neighbour").passed


def test_a_company_missing_from_the_search_index_fails(release):
    rewrite(root(release) / "search" / "docs.json", lambda d: d["rows"].pop())
    assert not gate(run_published_gates(release), "search index covers").passed


def test_a_manifest_without_provenance_fails(release):
    rewrite(root(release) / "manifest.json", lambda d: d.update({"limitations": [], "models": {}}))
    assert not gate(run_published_gates(release), "records versions").passed


def test_the_quality_report_is_covered_by_the_manifest(release):
    """Anything written after the checksum sweep records a stale hash.

    That failure surfaces on the *next* release, in a deploy job, which is a
    confusing place to learn about it. The quality report is written by the
    publisher for exactly this reason.
    """
    manifest = json.loads((root(release) / "manifest.json").read_text())
    published = set(manifest["checksums"])
    on_disk = {
        str(p.relative_to(root(release)))
        for p in root(release).rglob("*.json")
        if p.name != "manifest.json"
    }
    assert on_disk <= published, f"not checksummed: {sorted(on_disk - published)}"
