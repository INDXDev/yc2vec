"""Published artifact shape, transport keys, sharding and determinism."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.models import (
    Cluster,
    CompanyTagFeature,
    CompanyTagJudgment,
    EvidenceSpan,
    Neighbor,
    ReleaseManifest,
    SourceTaxonomyTagMapping,
    Tag,
    UmapPoint,
)
from pipeline.normalize.companies import normalize_companies
from pipeline.publish.browser import (
    COMPANY_KEY_MAP,
    DETAIL_SHARDS,
    PUBLISHED_NEIGHBORS_PER_SPACE,
    publish_browser_artifacts,
    shard_for,
)
from pipeline.publish.exports import write_exports
from pipeline.versions import PUBLIC_ARTIFACT_VERSION

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def tag(tag_id: str, facet: str = "industry") -> Tag:
    return Tag(
        tag_id=tag_id,
        canonical_name=tag_id.replace("-", " ").title(),
        definition=f"A definition of {tag_id} long enough to be usable.",
        facet=facet,
        state="active",
        support_count=3,
        source_company_ids=[],
        ontology_version="1",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def published(tmp_path: Path, sample_raws):
    companies = normalize_companies(sample_raws)[:16]
    ids = [c.company_id for c in companies]
    tags = [tag("alpha"), tag("beta", "buyer"), tag("gamma", "workflow")]

    judgments, features = [], []
    for i, cid in enumerate(ids):
        for t in tags[: (i % 3) + 1]:
            j = CompanyTagJudgment(
                judgment_id=f"{cid}:{t.tag_id}",
                company_id=cid,
                tag_id=t.tag_id,
                decision="yes",
                confidence=0.8,
                rationale="matches the definition",
                evidence=[EvidenceSpan(document_id=f"{cid}#yc_one_liner", quote="a quote")],
                shortlist_reason="retrieval",
                model="m",
                prompt_version="p",
                ontology_version="1",
                pipeline_version="1",
                run_id="r",
                created_at=NOW,
            )
            judgments.append(j)
            features.append(
                CompanyTagFeature(
                    company_id=cid,
                    tag_id=t.tag_id,
                    present=1,
                    raw_confidence=0.8,
                    calibrated_confidence=0.72,
                    information_weight=0.5,
                    feature_value=0.36,
                    judgment_id=j.judgment_id,
                    ontology_version="1",
                    run_id="r",
                )
            )
    # One more neighbour than the browser publishes, so the trim is exercised.
    neighbors = [
        Neighbor(
            company_id=ids[0],
            neighbor_company_id=ids[k],
            space="combined",
            rank=k - 1,
            similarity=0.9 - k * 0.01,
            embedding_space_version="v1",
        )
        for k in range(1, PUBLISHED_NEIGHBORS_PER_SPACE + 2)
    ]
    points = [
        UmapPoint(
            company_id=cid,
            x=i * 0.1,
            y=-i * 0.1,
            cluster_id=i % 2,
            projection_version="p1",
            embedding_space_version="v1",
        )
        for i, cid in enumerate(ids)
    ]
    clusters = [
        Cluster(
            cluster_id=c,
            label=f"Cluster {c}",
            size=6,
            top_tag_ids=["alpha"],
            centroid_x=0.0,
            centroid_y=0.0,
            projection_version="p1",
        )
        for c in (0, 1)
    ]
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
    )
    counts = publish_browser_artifacts(
        tmp_path,
        companies=companies,
        tags=tags,
        features=features,
        judgments=judgments,
        neighbors=neighbors,
        points=points,
        clusters=clusters,
        terms=[],
        mappings=[],
        manifest=manifest,
    )
    root = tmp_path / f"v{PUBLIC_ARTIFACT_VERSION}"
    return {"root": root, "counts": counts, "companies": companies, "ids": ids}


def load(root: Path, name: str):
    return json.loads((root / name).read_text())


def test_every_expected_artifact_is_written(published):
    root = published["root"]
    for name in (
        "manifest.json",
        "points.json",
        "companies.json",
        "tags.json",
        "taxonomy.json",
        "clusters.json",
        "search/docs.json",
    ):
        assert (root / name).exists(), name
    assert len(list((root / "detail").glob("*.json"))) == DETAIL_SHARDS


def test_points_arrays_are_parallel_and_finite(published):
    points = load(published["root"], "points.json")
    n = points["count"]
    for key in ("ids", "x", "y", "cluster", "year"):
        assert len(points[key]) == n, key
    assert all(isinstance(v, int | float) for v in points["x"] + points["y"])
    # The lossy-projection caveat travels with the data, not just the UI.
    assert "lossy" in points["note"].lower()


def test_company_rows_use_the_documented_key_map(published):
    rows = load(published["root"], "companies.json")["rows"]
    assert rows
    for row in rows:
        assert set(row) <= set(COMPANY_KEY_MAP), set(row) - set(COMPANY_KEY_MAP)
        assert row["i"] and row["n"]
        # Nulls and empties are dropped to keep the index small.
        assert all(v not in (None, "", []) for k, v in row.items())


def test_manifest_documents_the_key_map_and_shards(published):
    manifest = load(published["root"], "manifest.json")
    assert manifest["key_map"] == COMPANY_KEY_MAP
    assert manifest["detail_shards"] == DETAIL_SHARDS
    assert manifest["checksums"]
    assert "points.json" in manifest["checksums"]


def test_detail_records_are_reachable_at_their_shard(published):
    root, ids = published["root"], published["ids"]
    for cid in ids:
        shard = json.loads((root / "detail" / f"{shard_for(cid)}.json").read_text())
        assert cid in shard, f"{cid} missing from shard {shard_for(cid)}"
        record = shard[cid]
        assert record["company_id"] == cid
        assert record["coordinates"] is not None


def test_detail_tags_carry_evidence_and_rationale(published):
    root, ids = published["root"], published["ids"]
    record = json.loads((root / "detail" / f"{shard_for(ids[0])}.json").read_text())[ids[0]]
    assert record["tags"]
    for t in record["tags"]:
        assert t["rationale"]
        assert t["evidence"] and t["evidence"][0]["quote"]
        assert 0.0 <= t["confidence"] <= 1.0


def test_published_neighbours_are_trimmed_and_explained(published):
    root, ids = published["root"], published["ids"]
    record = json.loads((root / "detail" / f"{shard_for(ids[0])}.json").read_text())[ids[0]]
    combined = record["neighbors"]["combined"]
    assert len(combined) == PUBLISHED_NEIGHBORS_PER_SPACE
    # Ranking is preserved by the trim.
    assert combined == sorted(combined, key=lambda n: -n["score"])
    # "Why similar" is grounded in shared facts, never invented.
    assert any("shared_tags" in n or "shared_metadata" in n for n in combined)


def test_search_documents_cover_every_published_company(published):
    docs = load(published["root"], "search/docs.json")["rows"]
    assert {d["i"] for d in docs} == set(published["ids"])
    assert all(d["n"] for d in docs)


def test_tags_payload_reports_prevalence_and_cooccurrence(published):
    payload = load(published["root"], "tags.json")
    assert payload["count"] == len(payload["rows"])
    for row in payload["rows"]:
        # The whole active ontology ships, including tags with no assignments
        # yet; prevalence is reported honestly rather than filtered away.
        assert row["prevalence"] >= 0
        assert row["definition"]
    names = {r["tag_id"] for r in payload["rows"]}
    for row in payload["rows"]:
        assert all(c["tag_id"] in names for c in row["cooccurring"])


def test_publication_is_deterministic(tmp_path, sample_raws, published):
    """Byte-identical output for identical input, apart from the manifest timestamp."""
    import shutil

    first = tmp_path / "first"
    shutil.copytree(published["root"], first)
    # Re-run the same publication into the same directory.
    second = published["root"]
    for name in ("points.json", "companies.json", "tags.json", "search/docs.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_exports_write_csv_parquet_and_sparse_matrix(tmp_path, sample_raws):
    import numpy as np

    companies = normalize_companies(sample_raws)[:8]
    tags = [tag("alpha"), tag("beta", "buyer")]
    features = [
        CompanyTagFeature(
            company_id=c.company_id,
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
        for c in companies
    ]
    counts = write_exports(
        tmp_path,
        companies=companies,
        tags=tags,
        terms=[],
        mappings=[
            SourceTaxonomyTagMapping(
                mapping_id="m",
                term_id="industry:b2b",
                tag_id="alpha",
                relation="narrower",
                similarity=0.8,
                method="embedding",
                created_at=NOW,
            )
        ],
        features=features,
        judgments=[],
        neighbors=[],
        points=[],
    )
    assert counts["companies"] == 8
    assert (tmp_path / "companies.csv").exists()
    assert (tmp_path / "companies.parquet").exists()
    # The long-form edge table is the canonical company x tag export.
    assert (tmp_path / "company_tags.csv").exists()
    assert counts["matrix_nnz"] == 8

    with np.load(tmp_path / "company_tag_matrix.npz", allow_pickle=True) as data:
        assert tuple(data["shape"]) == (8, 2)
        assert len(data["company_ids"]) == 8
        assert list(data["tag_ids"]) == ["alpha", "beta"]


def test_csv_export_has_no_nested_python_reprs(tmp_path, sample_raws):
    companies = normalize_companies(sample_raws)[:5]
    write_exports(
        tmp_path,
        companies=companies,
        tags=[],
        terms=[],
        mappings=[],
        features=[],
        judgments=[],
        neighbors=[],
        points=[],
    )
    text = (tmp_path / "companies.csv").read_text()
    assert "['" not in text and "{'" not in text


# -- accessibility and honesty of the published copy -----------------------------


def test_published_disclaimers_are_present(published):
    """The lossy-projection and algorithmic-cluster caveats must ship with the data,
    so they survive even if someone consumes the artifacts without the UI."""
    points = load(published["root"], "points.json")
    clusters = load(published["root"], "clusters.json")
    assert "lossy" in points["note"].lower()
    assert "neighbour" in points["note"].lower()
    assert "not official" in clusters["disclaimer"].lower()


def test_manifest_records_limitations_and_attribution(tmp_path, sample_raws):
    """A consumer of the raw artifacts must be able to see what they can trust."""
    from pipeline.normalize.companies import normalize_companies

    companies = normalize_companies(sample_raws)[:4]
    manifest = ReleaseManifest(
        dataset_version="t",
        schema_version="1",
        public_artifact_version=PUBLIC_ARTIFACT_VERSION,
        pipeline_version="1",
        ontology_version="1",
        embedding_space_version="v1",
        projection_version="p1",
        generated_at=NOW,
        source_url="https://example.com",
        limitations=["tags are inferred"],
        attribution="unofficial project",
    )
    publish_browser_artifacts(
        tmp_path,
        companies=companies,
        tags=[],
        features=[],
        judgments=[],
        neighbors=[],
        points=[
            UmapPoint(
                company_id=c.company_id,
                x=0.0,
                y=0.0,
                cluster_id=0,
                projection_version="p1",
                embedding_space_version="v1",
            )
            for c in companies
        ],
        clusters=[],
        terms=[],
        mappings=[],
        manifest=manifest,
    )
    written = json.loads((tmp_path / f"v{PUBLIC_ARTIFACT_VERSION}" / "manifest.json").read_text())
    assert written["limitations"]
    assert written["attribution"]
    assert written["source_url"]


def test_companies_without_a_projection_are_not_published(tmp_path, sample_raws):
    """The map, the index and the detail shards must never disagree about who exists."""
    from pipeline.normalize.companies import normalize_companies

    companies = normalize_companies(sample_raws)[:6]
    projected = companies[:4]
    manifest = ReleaseManifest(
        dataset_version="t",
        schema_version="1",
        public_artifact_version=PUBLIC_ARTIFACT_VERSION,
        pipeline_version="1",
        ontology_version="1",
        embedding_space_version="v1",
        projection_version="p1",
        generated_at=NOW,
        source_url="https://example.com",
    )
    publish_browser_artifacts(
        tmp_path,
        companies=companies,
        tags=[],
        features=[],
        judgments=[],
        neighbors=[],
        points=[
            UmapPoint(
                company_id=c.company_id,
                x=0.0,
                y=0.0,
                cluster_id=0,
                projection_version="p1",
                embedding_space_version="v1",
            )
            for c in projected
        ],
        clusters=[],
        terms=[],
        mappings=[],
        manifest=manifest,
    )
    root = tmp_path / f"v{PUBLIC_ARTIFACT_VERSION}"
    published_ids = {r["i"] for r in json.loads((root / "companies.json").read_text())["rows"]}
    assert published_ids == {c.company_id for c in projected}
    assert json.loads((root / "points.json").read_text())["count"] == 4


def test_active_tags_without_assignments_are_still_published(tmp_path, sample_raws):
    """The ontology is a result in its own right; hiding the unassigned part of
    it would misrepresent both what was discovered and how far assignment got."""
    from pipeline.normalize.companies import normalize_companies

    companies = normalize_companies(sample_raws)[:4]
    tags = [tag("assigned"), tag("never-assigned")]
    judgment = CompanyTagJudgment(
        judgment_id="j",
        company_id=companies[0].company_id,
        tag_id="assigned",
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
        company_id=companies[0].company_id,
        tag_id="assigned",
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
        dataset_version="t",
        schema_version="1",
        public_artifact_version=PUBLIC_ARTIFACT_VERSION,
        pipeline_version="1",
        ontology_version="1",
        embedding_space_version="v1",
        projection_version="p1",
        generated_at=NOW,
        source_url="https://example.com",
    )
    publish_browser_artifacts(
        tmp_path,
        companies=companies,
        tags=tags,
        features=[feature],
        judgments=[judgment],
        neighbors=[],
        points=[
            UmapPoint(
                company_id=c.company_id,
                x=0.0,
                y=0.0,
                cluster_id=0,
                projection_version="p1",
                embedding_space_version="v1",
            )
            for c in companies
        ],
        clusters=[],
        terms=[],
        mappings=[],
        manifest=manifest,
    )
    rows = json.loads((tmp_path / f"v{PUBLIC_ARTIFACT_VERSION}" / "tags.json").read_text())["rows"]
    by_id = {r["tag_id"]: r for r in rows}
    assert set(by_id) == {"assigned", "never-assigned"}
    assert by_id["assigned"]["prevalence"] == 1
    assert by_id["never-assigned"]["prevalence"] == 0
    # Most-used first, so the useful tags surface even in a large ontology.
    assert rows[0]["tag_id"] == "assigned"
