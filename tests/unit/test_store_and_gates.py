"""Incremental invalidation, resume behaviour, and the release gates."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.models import (
    CompanyTagFeature,
    CompanyTagJudgment,
    EvidenceSpan,
    Neighbor,
    Tag,
    UmapPoint,
)
from pipeline.normalize.companies import normalize_companies
from pipeline.quality.evaluation import evaluate_dataset
from pipeline.quality.gates import run_release_gates
from pipeline.store import Store
from pipeline.util import append_jsonl, atomic_write, read_jsonl, write_jsonl

NOW = datetime(2026, 1, 1, tzinfo=UTC)


# -- store ------------------------------------------------------------------


def test_stage_key_changes_with_any_input(tmp_path):
    store = Store(tmp_path)
    base = store.stage_key("embed", {"model": "m", "prompt": "p"})
    assert base == store.stage_key("embed", {"model": "m", "prompt": "p"})
    assert base != store.stage_key("embed", {"model": "m2", "prompt": "p"})
    assert base != store.stage_key("embed", {"model": "m", "prompt": "p2"})
    assert base != store.stage_key("project", {"model": "m", "prompt": "p"})


def test_fresh_stage_is_skipped_and_stale_one_is_not(tmp_path):
    store = Store(tmp_path)
    out = store.path("inferred", "thing.jsonl")
    out.write_text("{}\n")
    key = store.stage_key("embed", {"v": 1})

    assert store.is_fresh("embed", key) is False
    store.record("embed", key, [out], {"rows": 1})
    assert store.is_fresh("embed", key) is True
    # A changed input produces a different key, so the stage reruns.
    assert store.is_fresh("embed", store.stage_key("embed", {"v": 2})) is False


def test_missing_output_invalidates_a_recorded_stage(tmp_path):
    store = Store(tmp_path)
    out = store.path("inferred", "thing.jsonl")
    out.write_text("{}\n")
    key = store.stage_key("embed", {"v": 1})
    store.record("embed", key, [out])
    out.unlink()
    assert store.is_fresh("embed", key) is False


def test_manifest_survives_a_reopen(tmp_path):
    store = Store(tmp_path)
    out = store.path("inferred", "thing.jsonl")
    out.write_text("{}\n")
    key = store.stage_key("embed", {"v": 1})
    store.record("embed", key, [out], {"rows": 7})
    assert Store(tmp_path).is_fresh("embed", key) is True
    assert Store(tmp_path).counts("embed") == {"rows": 7}


def test_invalidate_forces_a_rerun(tmp_path):
    store = Store(tmp_path)
    key = store.stage_key("embed", {"v": 1})
    store.record("embed", key)
    store.invalidate(["embed"])
    assert store.is_fresh("embed", key) is False


def test_cache_round_trips(tmp_path):
    store = Store(tmp_path)
    assert store.cache_get("ns", "key") is None
    store.cache_put("ns", "key", {"a": [1, 2]})
    assert store.cache_get("ns", "key") == {"a": [1, 2]}


# -- atomic IO and resume ----------------------------------------------------


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    target = tmp_path / "out.json"
    target.write_bytes(b'{"good": true}')
    with pytest.raises(RuntimeError), atomic_write(target) as fh:
        fh.write(b"garbage")
        raise RuntimeError("interrupted mid-write")
    # The previous release must survive an interrupted write.
    assert target.read_bytes() == b'{"good": true}'
    assert not list(tmp_path.glob("*.tmp*"))


def test_append_jsonl_checkpoints_incrementally(tmp_path):
    path = tmp_path / "partial.jsonl"
    append_jsonl(path, [{"company_id": "a"}])
    append_jsonl(path, [{"company_id": "b"}, {"company_id": "c"}])
    rows = list(read_jsonl(path))
    assert [r["company_id"] for r in rows] == ["a", "b", "c"]


def test_resume_skips_already_processed_companies(tmp_path):
    """The resume contract the assign stage relies on."""
    path = tmp_path / "partial.jsonl"
    append_jsonl(path, [{"company_id": "a", "tag_id": "t1"}, {"company_id": "a", "tag_id": "t2"}])
    done = {r["company_id"] for r in read_jsonl(path)}
    todo = [c for c in ["a", "b", "c"] if c not in done]
    assert todo == ["b", "c"]


def test_jsonl_write_is_deterministic(tmp_path):
    rows = [{"b": 2, "a": 1}, {"a": 3, "b": 4}]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, rows)
    write_jsonl(b, rows)
    assert a.read_bytes() == b.read_bytes()


# -- gates --------------------------------------------------------------------


def tag(tag_id: str, state: str = "active") -> Tag:
    return Tag(
        tag_id=tag_id,
        canonical_name=tag_id,
        definition="A definition long enough to be usable.",
        facet="industry",
        state=state,
        ontology_version="1",
        created_at=NOW,
        updated_at=NOW,
    )


def judgment(
    company_id: str, tag_id: str, *, evidence=True, rationale="because"
) -> CompanyTagJudgment:
    return CompanyTagJudgment(
        judgment_id=f"{company_id}:{tag_id}",
        company_id=company_id,
        tag_id=tag_id,
        decision="yes",
        confidence=0.9,
        rationale=rationale,
        evidence=[EvidenceSpan(document_id=f"{company_id}#d", quote="q")] if evidence else [],
        shortlist_reason="retrieval",
        model="m",
        prompt_version="p",
        ontology_version="1",
        pipeline_version="1",
        run_id="r",
        created_at=NOW,
    )


def feature(company_id: str, tag_id: str, value: float = 0.5) -> CompanyTagFeature:
    return CompanyTagFeature(
        company_id=company_id,
        tag_id=tag_id,
        present=1,
        raw_confidence=0.9,
        calibrated_confidence=0.9,
        information_weight=0.6,
        feature_value=value,
        judgment_id=f"{company_id}:{tag_id}",
        ontology_version="1",
        run_id="r",
    )


@pytest.fixture
def world(sample_raws):
    companies = normalize_companies(sample_raws)[:6]
    ids = [c.company_id for c in companies]
    return {
        "companies": companies,
        "tags": [tag("alpha"), tag("beta")],
        "features": [feature(ids[0], "alpha"), feature(ids[1], "beta")],
        "judgments": [judgment(ids[0], "alpha"), judgment(ids[1], "beta")],
        "neighbors": [
            Neighbor(
                company_id=ids[0],
                neighbor_company_id=ids[1],
                space="combined",
                rank=0,
                similarity=0.8,
                embedding_space_version="v1",
            )
        ],
        "points": [
            UmapPoint(
                company_id=c.company_id,
                x=float(i),
                y=float(-i),
                cluster_id=0,
                projection_version="p1",
                embedding_space_version="v1",
            )
            for i, c in enumerate(companies)
        ],
    }


def gate(results, name_fragment):
    return next(r for r in results if name_fragment in r.name)


def test_clean_dataset_passes_every_gate(world):
    assert all(r.passed for r in run_release_gates(**world))


def test_missing_provenance_fails(world):
    ids = [c.company_id for c in world["companies"]]
    world["judgments"][0] = judgment(ids[0], "alpha", evidence=False)
    assert gate(run_release_gates(**world), "provenance").passed is False


def test_missing_rationale_fails(world):
    ids = [c.company_id for c in world["companies"]]
    world["judgments"][0] = judgment(ids[0], "alpha", rationale="")
    assert gate(run_release_gates(**world), "provenance").passed is False


def test_nan_feature_value_fails(world):
    world["features"][0].feature_value = float("nan")
    assert gate(run_release_gates(**world), "NaN").passed is False


def test_infinite_coordinate_fails(world):
    world["points"][0].x = float("inf")
    assert gate(run_release_gates(**world), "NaN").passed is False


def test_dangling_neighbor_fails(world):
    world["neighbors"][0].neighbor_company_id = "ycoss:does-not-exist"
    assert gate(run_release_gates(**world), "neighbour").passed is False


def test_mixed_embedding_space_versions_fail(world):
    world["neighbors"].append(
        Neighbor(
            company_id=world["companies"][1].company_id,
            neighbor_company_id=world["companies"][0].company_id,
            space="combined",
            rank=0,
            similarity=0.7,
            embedding_space_version="v2",
        )
    )
    assert gate(run_release_gates(**world), "neighbour").passed is False


def test_inactive_tag_assignment_fails(world):
    world["tags"][0] = tag("alpha", state="merged")
    assert gate(run_release_gates(**world), "active tags only").passed is False


def test_dangling_company_reference_fails(world):
    world["features"].append(feature("ycoss:nope", "alpha"))
    assert gate(run_release_gates(**world), "existing company and tag").passed is False


def test_public_artifact_gate_detects_a_checksum_mismatch(world, tmp_path):
    import orjson

    from pipeline.versions import PUBLIC_ARTIFACT_VERSION

    root = tmp_path / f"v{PUBLIC_ARTIFACT_VERSION}"
    root.mkdir(parents=True)
    (root / "points.json").write_bytes(b'{"count": 1}')
    (root / "manifest.json").write_bytes(
        orjson.dumps({"checksums": {"points.json": "deadbeefdeadbeef"}})
    )
    result = run_release_gates(**world, public_dir=tmp_path)
    assert gate(result, "public artifacts").passed is False


# -- evaluation ----------------------------------------------------------------


def test_evaluation_reports_honest_metrics(world):
    metrics = evaluate_dataset(
        companies_count=len(world["companies"]),
        tags=world["tags"],
        features=world["features"],
        judgments=world["judgments"],
    )
    assert metrics["active_tags"] == 2
    assert metrics["evidence_coverage"] == 1.0
    assert metrics["orphan_tag_rate"] == 0.0
    assert metrics["duplicate_tag_rate"] == 0.0


def test_evaluation_flags_orphan_tags(world):
    world["tags"].append(tag("never-assigned"))
    metrics = evaluate_dataset(
        companies_count=6,
        tags=world["tags"],
        features=world["features"],
        judgments=world["judgments"],
    )
    assert "never-assigned" in metrics["orphan_tags"]
    assert metrics["orphan_tag_rate"] > 0


def test_evaluation_scores_against_a_gold_set(world, tmp_path):
    import json

    ids = [c.company_id for c in world["companies"]]
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "judgments": [
                    {"company_id": ids[0], "tag_id": "alpha", "decision": "yes"},
                    {"company_id": ids[1], "tag_id": "beta", "decision": "no"},
                ]
            }
        )
    )
    metrics = evaluate_dataset(
        companies_count=6,
        tags=world["tags"],
        features=world["features"],
        judgments=world["judgments"],
        gold_path=Path(gold),
    )
    sample = metrics["reviewed_sample"]
    assert sample["matched_pairs"] == 2
    assert sample["precision"] == pytest.approx(0.5)  # one true positive, one false positive
