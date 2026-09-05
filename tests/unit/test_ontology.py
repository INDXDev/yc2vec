"""Tag ids, aliasing, merge migrations and activation rules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pipeline.models import MergeProposal
from pipeline.ontology.discovery import _display_name, _to_candidate, diverse_batches
from pipeline.ontology.merge import activate_candidates, apply_merge
from pipeline.ontology.registry import OntologyRegistry
from pipeline.util import normalize_name

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def registry(tmp_path) -> OntologyRegistry:
    return OntologyRegistry(tmp_path / "ontology")


def make(registry: OntologyRegistry, name: str, facet: str = "industry", support=("a", "b")):
    return registry.create_tag(
        name=name,
        facet=facet,
        definition=f"{name} is a reusable attribute with a sufficiently long definition.",
        support_company_ids=support,
        created_at=NOW,
    )


def test_tag_id_is_stable_across_renames(registry):
    tag = make(registry, "AI Agents")
    original = tag.tag_id
    tag.canonical_name = "Autonomous Agents"
    registry.save()

    reloaded = OntologyRegistry(registry.root)
    assert reloaded.tags[original].tag_id == original
    assert reloaded.tags[original].canonical_name == "Autonomous Agents"


def test_ids_are_unique_under_collision(registry):
    a = make(registry, "Analytics", facet="industry")
    b = make(registry, "Analytics", facet="technology")
    assert a.tag_id != b.tag_id


def test_alias_resolution_is_normalisation_insensitive(registry):
    tag = make(registry, "AI Agents")
    for variant in ("ai agents", "AI  Agents", "AI-Agent", "ai_agent"):
        assert registry.resolve(variant) is not None
        assert registry.resolve(variant).tag_id == tag.tag_id


def test_normalize_name_collapses_variants():
    forms = ["AI Agents", "ai-agent", "AI  agent", "the AI agents"]
    assert len({normalize_name(f) for f in forms}) == 1
    # It must not collapse genuinely different names.
    assert normalize_name("data pipeline") != normalize_name("data warehouse")


def test_merge_is_a_migration_not_a_deletion(registry):
    keep = make(registry, "Developer Tools", support=("a", "b", "c"))
    drop = make(registry, "Devtools", support=("d",))
    proposal = MergeProposal(
        proposal_id="p",
        source_tag_id=drop.tag_id,
        target_tag_id=keep.tag_id,
        similarity=0.97,
        verdict="auto_merge",
        adjudication="merge",
        created_at=NOW,
    )
    assert apply_merge(registry, proposal) is True

    # The losing tag survives with a pointer, so old judgments still resolve.
    assert registry.tags[drop.tag_id].state == "merged"
    assert registry.tags[drop.tag_id].merged_into == keep.tag_id
    assert registry.follow(drop.tag_id).tag_id == keep.tag_id
    # Its name becomes an alias of the survivor, and support accumulates.
    assert registry.resolve("Devtools").tag_id == keep.tag_id
    assert set(registry.tags[keep.tag_id].source_company_ids) == {"a", "b", "c", "d"}


def test_merge_is_idempotent(registry):
    keep, drop = make(registry, "A Tag"), make(registry, "B Tag")
    p = MergeProposal(
        proposal_id="p",
        source_tag_id=drop.tag_id,
        target_tag_id=keep.tag_id,
        similarity=0.99,
        verdict="auto_merge",
        adjudication="merge",
        created_at=NOW,
    )
    assert apply_merge(registry, p) is True
    assert apply_merge(registry, p) is False


def test_unapproved_merges_are_never_applied(registry):
    keep, drop = make(registry, "A Tag"), make(registry, "B Tag")
    for adjudication in (None, "distinct", "unclear"):
        p = MergeProposal(
            proposal_id="p",
            source_tag_id=drop.tag_id,
            target_tag_id=keep.tag_id,
            similarity=0.9,
            verdict="review",
            adjudication=adjudication,
            created_at=NOW,
        )
        assert apply_merge(registry, p) is False
    assert registry.tags[drop.tag_id].state == "candidate"


def test_merge_chains_resolve(registry):
    a, b, c = make(registry, "One"), make(registry, "Two"), make(registry, "Three")
    for src, dst in ((c, b), (b, a)):
        apply_merge(
            registry,
            MergeProposal(
                proposal_id=f"{src.tag_id}",
                source_tag_id=src.tag_id,
                target_tag_id=dst.tag_id,
                similarity=0.99,
                verdict="auto_merge",
                adjudication="merge",
                created_at=NOW,
            ),
        )
    assert registry.follow(c.tag_id).tag_id == a.tag_id


def test_activation_requires_support_and_a_real_definition(registry):
    weak = make(registry, "Weak Tag", support=("a",))
    strong = make(registry, "Strong Tag", support=("a", "b", "c"))
    thin = registry.create_tag(name="Thin", facet="industry", definition="short", created_at=NOW)

    assert activate_candidates(registry, min_support=3) == 1
    assert registry.tags[strong.tag_id].state == "active"
    assert registry.tags[weak.tag_id].state == "candidate"
    assert registry.tags[thin.tag_id].state == "candidate"

    # An explicit override lets a rare but meaningful tag through.
    assert registry.activate(weak.tag_id, min_support=3, override=True) is True


def test_registry_round_trips(registry):
    make(registry, "Round Trip")
    registry.add_alias(registry.resolve("Round Trip").tag_id, "RT")
    registry.save()
    reloaded = OntologyRegistry(registry.root)
    assert reloaded.resolve("RT") is not None
    assert reloaded.stats()["total"] == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("b2b_software_subscription", "B2B Software Subscription"),
        ("AI-Agents", "AI Agents"),
        ("developer tools", "Developer Tools"),
        ("SaaS", "SaaS"),
    ],
)
def test_display_names_are_readable(raw, expected):
    assert _display_name(raw) == expected


def test_candidate_drops_support_ids_outside_the_batch():
    """The model sometimes cites companies it was never shown."""
    facets = ("industry", "business_model")
    item = {
        "name": "Made Up",
        "facet": "industry",
        "definition": "A definition that is comfortably long enough to pass validation.",
        "supporting_company_ids": ["c1", "hallucinated-id"],
    }
    candidate = _to_candidate(item, facets, {"c1"}, "run", "m")
    assert candidate is not None
    assert candidate.support_company_ids == ["c1"]

    # With no surviving support there is nothing to justify the candidate.
    item["supporting_company_ids"] = ["hallucinated-id"]
    assert _to_candidate(item, facets, {"c1"}, "run", "m") is None


def test_candidate_snaps_near_miss_facets_and_rejects_unmappable_ones():
    facets = ("industry", "business_model")
    base = {
        "name": "Made Up",
        "definition": "A definition that is comfortably long enough to pass validation.",
        "supporting_company_ids": ["c1"],
    }
    # A near miss is snapped onto the controlled list...
    snapped = _to_candidate({**base, "facet": "business model"}, facets, {"c1"}, "run", "m")
    assert snapped is not None and snapped.facet == "business_model"

    # ...but an invented facet with nothing in common is rejected outright,
    # rather than being filed under an arbitrary neighbour.
    assert _to_candidate({**base, "facet": "vibes"}, facets, {"c1"}, "run", "m") is None


def test_diverse_batches_mix_industries(sample_raws):
    from pipeline.normalize.companies import normalize_companies

    companies = normalize_companies(sample_raws)
    batches = diverse_batches(companies, 6)
    assert sum(len(b) for b in batches) == len(companies)
    # Every company appears exactly once.
    assert len({c.company_id for b in batches for c in b}) == len(companies)
    # Batches should not be single-industry blocks.
    multi = [b for b in batches if len({c.industry for c in b}) > 1]
    assert len(multi) >= len(batches) // 2
