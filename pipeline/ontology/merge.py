"""Stages 2-4: normalisation, merge review and activation.

Merging is the part of an open-ended ontology that quietly destroys data if it
goes wrong, so the rules are explicit and three-tiered:

* similarity >= ``auto_merge_threshold``  -> deterministic auto-merge
* ``review_threshold`` <= sim < auto      -> LLM adjudication; ``unclear`` verdicts
                                             land in a review queue and are never
                                             applied automatically
* similarity < ``review_threshold``       -> treated as distinct

Merges are applied as migrations: the losing tag stays in the registry with
``state='merged'`` and ``merged_into`` set, so historical judgments still
resolve.
"""

from __future__ import annotations

import asyncio

import numpy as np

from pipeline.config import Config
from pipeline.models import MergeProposal, Tag
from pipeline.ollama import OllamaClient
from pipeline.ontology.registry import OntologyRegistry
from pipeline.prompts import MERGE_SCHEMA, MERGE_SYSTEM, PROMPT_VERSIONS, merge_prompt
from pipeline.util import log, normalize_name, now

LOG = log(__name__)


def tag_document(tag: Tag) -> str:
    """The text embedded to represent a tag's meaning."""
    return f"{tag.canonical_name} ({tag.facet}): {tag.definition}"


async def embed_tags(client: OllamaClient, tags: list[Tag], batch_size: int = 16) -> np.ndarray:
    """L2-normalised definition embeddings, in the order given."""
    if not tags:
        return np.zeros((0, 0), dtype=np.float32)
    vectors: list[list[float]] = []
    docs = [tag_document(t) for t in tags]
    for i in range(0, len(docs), batch_size):
        vectors.extend(await client.embed(docs[i : i + batch_size]))
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


async def propose_merges(
    *,
    config: Config,
    client: OllamaClient,
    registry: OntologyRegistry,
    tags: list[Tag],
    adjudicate: bool = True,
    max_adjudications: int | None = None,
    concurrency: int = 4,
) -> list[MergeProposal]:
    """Compare every tag pair within a facet and classify the relationship.

    An open-ended ontology produces far more borderline pairs than it is
    sensible to spend model time on, so adjudication is bounded: the most
    similar pairs (the ones most likely to be true duplicates) are judged, and
    the remainder stay in the review queue as unresolved proposals. Nothing in
    the queue is ever merged automatically.
    """
    if len(tags) < 2:
        return []
    matrix = await embed_tags(client, tags, config.embeddings.batch_size)
    sims = matrix @ matrix.T

    cfg = config.ontology
    proposals: list[MergeProposal] = []
    pending: list[tuple[int, int, float]] = []

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            # Cross-facet duplicates are legitimate (the same word can be an
            # industry and a technology), so only compare within a facet.
            if tags[i].facet != tags[j].facet:
                continue
            sim = float(sims[i, j])
            # Identical normalised names are the same tag regardless of the
            # embedding score.
            same_name = normalize_name(tags[i].canonical_name) == normalize_name(
                tags[j].canonical_name
            )
            if sim >= cfg.auto_merge_threshold or same_name:
                proposals.append(
                    _proposal(tags[i], tags[j], sim, "auto_merge", adjudication="merge")
                )
            elif sim >= cfg.review_threshold:
                pending.append((i, j, sim))

    # Judge the most similar pairs first; they carry the most merge signal per call.
    pending.sort(key=lambda t: -t[2])
    deferred: list[tuple[int, int, float]] = []
    if max_adjudications is not None and len(pending) > max_adjudications:
        deferred = pending[max_adjudications:]
        pending = pending[:max_adjudications]
        LOG.info(
            "merge review: adjudicating the %d most similar pairs, %d left in the review queue",
            len(pending),
            len(deferred),
        )

    if pending and adjudicate:
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int, j: int, sim: float) -> MergeProposal:
            a, b = tags[i], tags[j]
            async with sem:
                try:
                    res = await client.generate_json(
                        system=MERGE_SYSTEM,
                        prompt=merge_prompt(
                            {
                                "name": a.canonical_name,
                                "facet": a.facet,
                                "definition": a.definition,
                            },
                            {
                                "name": b.canonical_name,
                                "facet": b.facet,
                                "definition": b.definition,
                            },
                            sim,
                        ),
                        schema=MERGE_SCHEMA,
                        prompt_version=PROMPT_VERSIONS["merge"],
                        num_predict=300,
                        cache_namespace="merge",
                    )
                except Exception as exc:
                    LOG.warning("merge adjudication failed for %s/%s: %s", a.tag_id, b.tag_id, exc)
                    return _proposal(
                        a, b, sim, "review", adjudication=None, rationale=str(exc)[:200]
                    )
            return _proposal(
                a,
                b,
                sim,
                "review",
                adjudication=res.get("verdict"),
                rationale=res.get("rationale"),
                model=config.models.chat_model,
            )

        proposals.extend(await asyncio.gather(*(one(i, j, s) for i, j, s in pending)))
    elif pending:
        deferred = deferred + pending
    proposals.extend(_proposal(tags[i], tags[j], s, "review") for i, j, s in deferred)

    LOG.info(
        "merge review: %d auto, %d adjudicated (%d merge, %d distinct, %d unclear)",
        sum(1 for p in proposals if p.verdict == "auto_merge"),
        sum(1 for p in proposals if p.verdict == "review"),
        sum(1 for p in proposals if p.verdict == "review" and p.adjudication == "merge"),
        sum(1 for p in proposals if p.verdict == "review" and p.adjudication == "distinct"),
        sum(
            1
            for p in proposals
            if p.verdict == "review" and p.adjudication not in ("merge", "distinct")
        ),
    )
    return proposals


def _proposal(
    a: Tag,
    b: Tag,
    sim: float,
    verdict: str,
    *,
    adjudication: str | None = None,
    rationale: str | None = None,
    model: str | None = None,
) -> MergeProposal:
    # The better-supported tag survives; ties break on id so the outcome is
    # deterministic across runs.
    if (a.support_count, b.tag_id) >= (b.support_count, a.tag_id):
        target, source = a, b
    else:
        target, source = b, a
    return MergeProposal(
        proposal_id=f"{source.tag_id}->{target.tag_id}",
        source_tag_id=source.tag_id,
        target_tag_id=target.tag_id,
        similarity=sim,
        verdict=verdict,  # type: ignore[arg-type]
        adjudication=adjudication,  # type: ignore[arg-type]
        rationale=rationale,
        model=model,
        created_at=now(),
    )


def apply_merge(registry: OntologyRegistry, proposal: MergeProposal) -> bool:
    """Apply an approved merge as a migration. Returns False if not applicable."""
    if proposal.applied:
        return False
    if proposal.verdict != "auto_merge" and proposal.adjudication != "merge":
        return False
    source = registry.tags.get(proposal.source_tag_id)
    target = registry.follow(proposal.target_tag_id)
    if source is None or target is None or source.tag_id == target.tag_id:
        return False
    if source.state == "merged":
        return False

    registry.add_alias(target.tag_id, source.canonical_name, origin="merge")
    for alias in source.aliases:
        registry.add_alias(target.tag_id, alias, origin="merge")
    target.source_company_ids = sorted(
        set(target.source_company_ids) | set(source.source_company_ids)
    )
    target.support_count = len(target.source_company_ids)
    target.positive_examples = list(
        dict.fromkeys([*target.positive_examples, *source.positive_examples])
    )[:6]
    target.updated_at = now()

    source.state = "merged"
    source.merged_into = target.tag_id
    source.updated_at = now()
    proposal.applied = True
    registry._reindex()
    return True


def activate_candidates(
    registry: OntologyRegistry,
    *,
    min_support: int,
    overrides: set[str] | None = None,
) -> int:
    """Promote every candidate tag that clears the activation rules."""
    overrides = overrides or set()
    activated = 0
    for tag in list(registry.tags.values()):
        if tag.state != "candidate":
            continue
        if registry.activate(tag.tag_id, min_support=min_support, override=tag.tag_id in overrides):
            activated += 1
    return activated
