"""Source parsing, normalisation and schema-drift behaviour."""

from __future__ import annotations

import pytest

from pipeline.adapters.yc_oss_api import YcOssApiAdapter
from pipeline.normalize.companies import (
    build_metadata_document,
    company_documents,
    normalize_companies,
    normalize_company,
    parse_batch,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Winter 2012", ("Winter", 2012, "winter-2012")),
        ("Summer 2005", ("Summer", 2005, "summer-2005")),
        ("Spring 2024", ("Spring", 2024, "spring-2024")),
        ("Fall 2024", ("Fall", 2024, "fall-2024")),
        (None, ("Unspecified", None, None)),
        # Historical labels must keep their exact source string, not be guessed at.
        ("IK12", ("Unspecified", None, "ik12")),
        ("Unspecified", ("Unspecified", None, "unspecified")),
    ],
)
def test_parse_batch(value, expected):
    assert parse_batch(value) == expected


def test_normalize_preserves_source_classifications(sample_raws):
    companies = normalize_companies(sample_raws)
    assert len(companies) == len(sample_raws)
    for c, raw in zip(companies, sample_raws, strict=True):
        assert c.company_id == raw.company_id
        assert c.name == raw.payload["name"]
        # Source names are reproduced exactly, never renamed.
        assert c.industry == (raw.payload.get("industry") or None)
        assert c.source_tags == [t for t in raw.payload.get("tags", []) if t]


def test_missing_metadata_stays_missing(sample_raws):
    raw = sample_raws[0].model_copy(deep=True)
    for field in ("team_size", "status", "stage", "industry", "all_locations"):
        raw.payload.pop(field, None)
    c = normalize_company(raw)
    assert c.team_size is None
    assert c.status is None
    assert c.stage is None
    assert c.industry is None
    # A missing field simply does not appear in the metadata document.
    assert "status is" not in c.metadata_document
    assert "team size" not in c.metadata_document.lower()


def test_unknown_source_field_does_not_break_parsing(sample_raws):
    """Schema drift upstream must not crash the pipeline."""
    raw = sample_raws[0].model_copy(deep=True)
    raw.payload["brand_new_upstream_field"] = {"nested": [1, 2, 3]}
    c = normalize_company(raw)
    assert c.name


def test_metadata_document_is_deterministic(sample_raws):
    c = normalize_company(sample_raws[0])
    assert build_metadata_document(c) == build_metadata_document(c)
    assert build_metadata_document(c) == c.metadata_document
    # It describes metadata, never the company's own prose.
    if c.long_description:
        assert c.long_description[:60] not in c.metadata_document


def test_content_hash_changes_only_with_content(sample_raws):
    a = normalize_company(sample_raws[0])
    raw = sample_raws[0].model_copy(deep=True)
    b = normalize_company(raw)
    assert a.content_hash == b.content_hash

    raw.payload["one_liner"] = "something entirely different"
    assert normalize_company(raw).content_hash != a.content_hash


def test_documents_have_stable_ids(sample_raws):
    c = normalize_company(sample_raws[0])
    docs = company_documents(c, website_text="Extracted homepage text about the product.")
    ids = [d.document_id for d in docs]
    assert len(ids) == len(set(ids))
    assert all(d.document_id.startswith(c.company_id + "#") for d in docs)
    assert any(d.kind == "website_main_text" for d in docs)


def test_derived_taxonomy_terms(sample_payloads):
    from datetime import UTC, datetime

    terms = YcOssApiAdapter.derived_terms(sample_payloads, datetime(2026, 1, 1, tzinfo=UTC))
    kinds = {t.kind for t in terms}
    assert {"subindustry", "region", "status"} <= kinds
    # Ids are stable and unique; display names keep their original form.
    assert len({t.term_id for t in terms}) == len(terms)
    sub = [t for t in terms if t.kind == "subindustry"]
    assert all(t.parent_term_id and t.parent_term_id.startswith("industry:") for t in sub)
