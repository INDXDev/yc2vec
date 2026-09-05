"""Shortlisting, judgment hygiene and the published feature value."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pipeline.embeddings.spaces import l2_normalize
from pipeline.models import CompanyTagJudgment, EvidenceSpan, SourceDocument, Tag
from pipeline.normalize.companies import normalize_company
from pipeline.tagging.assign import _verify_evidence, build_features, calibrate, information_weight
from pipeline.tagging.shortlist import Shortlister
from pipeline.versions import EXTRACTION_VERSION

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_tag(tag_id: str, facet: str, name: str | None = None) -> Tag:
    return Tag(
        tag_id=tag_id,
        canonical_name=name or tag_id.replace("-", " ").title(),
        definition=f"Definition of {tag_id} that is long enough to be usable.",
        facet=facet,
        normalized_aliases=[(name or tag_id).lower().replace("-", " ")],
        state="active",
        ontology_version="1",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def tags() -> list[Tag]:
    return [
        make_tag("developer-tools", "product_form", "developer tools"),
        make_tag("marketplace", "business_model"),
        make_tag("healthcare", "industry"),
        make_tag("enterprise-buyer", "buyer"),
        make_tag("logistics", "industry"),
        make_tag("computer-vision", "technology"),
        make_tag("subscription", "business_model"),
        make_tag("consumer-app", "product_form"),
    ]


def test_shortlist_covers_every_facet(tags, sample_raws):
    company = normalize_company(sample_raws[0])
    rng = np.random.default_rng(0)
    matrix = l2_normalize(rng.normal(size=(len(tags), 8)))
    vector = l2_normalize(rng.normal(size=8))

    items = Shortlister(tags, matrix, shortlist_size=3, hard_negatives=0).shortlist(
        company, vector, f"{company.name} {company.one_liner}"
    )
    picked = {i.tag_id for i in items}
    facets = {t.facet for t in tags if t.tag_id in picked}
    assert facets == {t.facet for t in tags}  # no facet is structurally ignored


def test_shortlist_includes_hard_negatives(tags, sample_raws):
    company = normalize_company(sample_raws[0])
    rng = np.random.default_rng(1)
    matrix = l2_normalize(rng.normal(size=(len(tags), 8)))
    vector = l2_normalize(rng.normal(size=8))
    items = Shortlister(tags, matrix, shortlist_size=2, hard_negatives=2, seed=5).shortlist(
        company, vector, company.name
    )
    assert any(i.reason == "hard_negative" for i in items)
    assert len({i.tag_id for i in items}) == len(items)  # no duplicates


def test_shortlist_is_deterministic(tags, sample_raws):
    company = normalize_company(sample_raws[0])
    rng = np.random.default_rng(2)
    matrix = l2_normalize(rng.normal(size=(len(tags), 8)))
    vector = l2_normalize(rng.normal(size=8))
    a = Shortlister(tags, matrix, hard_negatives=2, seed=9).shortlist(company, vector, company.name)
    b = Shortlister(tags, matrix, hard_negatives=2, seed=9).shortlist(company, vector, company.name)
    assert [(i.tag_id, i.reason) for i in a] == [(i.tag_id, i.reason) for i in b]


def test_alias_hits_are_shortlisted(tags):
    from pipeline.models import CompanyNormalized

    company = CompanyNormalized(
        company_id="c1",
        yc_id=1,
        slug="acme",
        name="Acme",
        one_liner="We build developer tools for backend teams.",
        yc_url="https://example.com",
        metadata_document="Acme is a Y Combinator company.",
        metadata_template_version="1",
        normalize_version="1",
        content_hash="h",
        updated_at=NOW,
    )
    matrix = l2_normalize(np.zeros((len(tags), 8)) + 0.01)
    items = Shortlister(tags, matrix, shortlist_size=1, hard_negatives=0).shortlist(
        company, l2_normalize(np.ones(8)), company.one_liner or ""
    )
    reasons = {i.tag_id: i.reason for i in items}
    assert reasons.get("developer-tools") in {"alias", "retrieval", "facet_prior"}
    assert "developer-tools" in reasons


def doc(text: str, doc_id: str = "c#d") -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        company_id="c",
        kind="yc_long_description",
        text=text,
        char_count=len(text),
        content_hash="h",
        extraction_version=EXTRACTION_VERSION,
        created_at=NOW,
    )


def test_evidence_must_actually_occur_in_the_document():
    documents = [doc("Acme builds industrial widgets for factory automation.")]
    kept = _verify_evidence(
        [
            {"document_id": "c#d", "quote": "builds industrial widgets"},  # real
            {"document_id": "c#d", "quote": "sells consumer software"},  # invented
            {"document_id": "c#missing", "quote": "builds industrial widgets"},  # wrong doc
        ],
        documents,
    )
    assert [e.quote for e in kept] == ["builds industrial widgets"]


def test_evidence_matching_tolerates_whitespace_and_case():
    documents = [doc("Acme  builds   INDUSTRIAL widgets.")]
    kept = _verify_evidence(
        [{"document_id": "c#d", "quote": "builds industrial widgets"}], documents
    )
    assert len(kept) == 1


def judgment(tag_id: str, decision: str, confidence: float, *, evidence=True, reason="retrieval"):
    return CompanyTagJudgment(
        judgment_id=f"j-{tag_id}-{decision}",
        company_id="c1",
        tag_id=tag_id,
        decision=decision,
        confidence=confidence,
        rationale="because",
        evidence=[EvidenceSpan(document_id="c1#d", quote="q")] if evidence else [],
        shortlist_reason=reason,
        model="m",
        prompt_version="p",
        ontology_version="1",
        pipeline_version="1",
        run_id="r",
        created_at=NOW,
    )


def test_only_confident_evidenced_positives_become_features():
    judgments = [
        judgment("a", "yes", 0.9),
        judgment("b", "yes", 0.3),  # below threshold
        judgment("c", "no", 0.95),  # negative
        judgment("d", "uncertain", 0.8),  # undecided
        judgment("e", "yes", 0.9, evidence=False),  # unprovenanced
    ]
    features = build_features(judgments, n_companies=10, min_confidence=0.55)
    assert {f.tag_id for f in features} == {"a"}


def test_no_and_uncertain_are_never_conflated():
    judgments = [judgment("x", "no", 0.9), judgment("y", "uncertain", 0.9)]
    assert build_features(judgments, n_companies=10, min_confidence=0.0) == []


def test_feature_value_is_confidence_times_information_weight():
    features = build_features([judgment("a", "yes", 0.9)], n_companies=100, min_confidence=0.5)
    f = features[0]
    assert f.present == 1
    assert f.raw_confidence == pytest.approx(0.9)
    assert f.calibrated_confidence == pytest.approx(calibrate(0.9, "yes"))
    assert f.feature_value == pytest.approx(
        round(f.calibrated_confidence * f.information_weight, 6)
    )
    assert f.feature_value != f.raw_confidence  # unweighted values are kept separately


def test_ubiquitous_tags_are_down_weighted():
    common = [
        judgment("common", "yes", 0.9).model_copy(
            update={"company_id": f"c{i}", "judgment_id": f"j{i}"}
        )
        for i in range(100)
    ]
    rare = [
        judgment("rare", "yes", 0.9).model_copy(update={"company_id": "c0", "judgment_id": "jr"})
    ]
    features = build_features(common + rare, n_companies=100, min_confidence=0.5)
    weights = {f.tag_id: f.information_weight for f in features}
    assert weights["rare"] > weights["common"]


def test_information_weight_bounds():
    assert information_weight(1.0) == pytest.approx(0.15, abs=1e-6)  # on everything
    assert information_weight(1e-7) == pytest.approx(1.0, abs=1e-6)  # vanishingly rare
    assert 0.15 <= information_weight(0.5) <= 1.0


def test_calibration_is_monotonic_and_zero_for_non_positives():
    values = [calibrate(c, "yes") for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
    assert calibrate(1.0, "yes") == pytest.approx(1.0)
    assert calibrate(0.99, "no") == 0.0
    assert calibrate(0.99, "uncertain") == 0.0


def test_duplicate_pairs_collapse_to_one_feature():
    a = judgment("t", "yes", 0.9)
    b = judgment("t", "yes", 0.8).model_copy(update={"judgment_id": "j2"})
    assert len(build_features([a, b], n_companies=5, min_confidence=0.5)) == 1
