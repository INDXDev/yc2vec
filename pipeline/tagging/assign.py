"""Stage 5: evidence-grounded company/tag judgments, and the sparse features.

Every shortlisted pair gets its own judgment record: decision, confidence,
rationale, evidence spans, and the model/prompt/ontology/run versions that
produced it. ``uncertain`` is a first-class outcome -- weak evidence is never
rounded to yes or no.

Calls are grouped by facet for throughput (``tagging.pairs_per_call``), which
does not change what is judged: the prompt asks for one independent judgment
object per tag and each is stored separately. Set ``pairs_per_call=1`` for
strictly one call per pair.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict

from pipeline.config import Config
from pipeline.models import (
    CompanyNormalized,
    CompanyTagFeature,
    CompanyTagJudgment,
    EvidenceSpan,
    SourceDocument,
    Tag,
)
from pipeline.ollama import OllamaClient
from pipeline.prompts import ASSIGN_SCHEMA, ASSIGN_SYSTEM, PROMPT_VERSIONS, assign_prompt
from pipeline.tagging.shortlist import ShortlistItem
from pipeline.util import log, normalize_name, now, stable_hash, truncate
from pipeline.versions import ONTOLOGY_VERSION

LOG = log(__name__)


def _judgment_id(company_id: str, tag_id: str, run_id: str) -> str:
    return stable_hash({"c": company_id, "t": tag_id, "r": run_id})[:20]


def _verify_evidence(quotes: list[dict], documents: list[SourceDocument]) -> list[EvidenceSpan]:
    """Keep only quotes that actually occur in the evidence we supplied.

    The model is asked for verbatim spans; anything it paraphrases or invents is
    dropped rather than published as provenance.
    """
    by_id = {d.document_id: normalize_name(d.text) for d in documents}
    spans: list[EvidenceSpan] = []
    for q in quotes or []:
        doc_id = str(q.get("document_id", ""))
        quote = " ".join(str(q.get("quote", "")).split())
        if not quote or doc_id not in by_id:
            continue
        if normalize_name(quote) in by_id[doc_id]:
            spans.append(EvidenceSpan(document_id=doc_id, quote=truncate(quote, 240)))
    return spans


async def assign_tags(
    *,
    config: Config,
    client: OllamaClient,
    company: CompanyNormalized,
    documents: list[SourceDocument],
    tags_by_id: dict[str, Tag],
    shortlist: list[ShortlistItem],
    run_id: str,
    model_digest: str | None = None,
) -> list[CompanyTagJudgment]:
    """Judge one company against its shortlist. Returns one record per pair."""
    from pipeline import PIPELINE_VERSION

    groups: dict[str, list[ShortlistItem]] = defaultdict(list)
    for item in shortlist:
        tag = tags_by_id.get(item.tag_id)
        if tag is not None:
            groups[tag.facet].append(item)

    # Facets keep related tags together, which makes each call's context
    # coherent. But with a large ontology a company's shortlist spreads thinly
    # across many facets, and one call per facet would mean a dozen calls each
    # judging two tags -- paying the full evidence prompt every time. Small
    # facet groups are therefore packed together up to `pairs_per_call`.
    # This changes only how the calls are batched: each tag still gets its own
    # independent decision, confidence, rationale and evidence.
    per_call = max(1, config.tagging.pairs_per_call)
    calls: list[list[ShortlistItem]] = []
    pending: list[ShortlistItem] = []
    for items in sorted(groups.values(), key=len, reverse=True):
        for i in range(0, len(items), per_call):
            chunk = items[i : i + per_call]
            if len(chunk) == per_call:
                calls.append(chunk)
            elif len(pending) + len(chunk) <= per_call:
                pending.extend(chunk)
            else:
                calls.append(pending)
                pending = list(chunk)
    if pending:
        calls.append(pending)

    judgments: list[CompanyTagJudgment] = []
    for chunk in calls:
        tag_payload = [
            {
                "tag_id": t.tag_id,
                "name": t.canonical_name,
                "definition": t.definition,
            }
            for t in (tags_by_id[i.tag_id] for i in chunk)
        ]
        doc_payload = [
            {"document_id": d.document_id, "kind": d.kind, "text": d.text} for d in documents
        ]
        try:
            result = await client.generate_json(
                system=ASSIGN_SYSTEM,
                prompt=assign_prompt(company.name, doc_payload, tag_payload),
                schema=ASSIGN_SCHEMA,
                prompt_version=PROMPT_VERSIONS["assign"],
                num_predict=220 * len(chunk) + 200,
                cache_namespace="assign",
            )
        except Exception as exc:
            LOG.warning(
                "assignment failed for %s (%d tags): %s", company.company_id, len(chunk), exc
            )
            continue

        by_tag = {str(j.get("tag_id")): j for j in result.get("judgments", [])}
        reasons = {i.tag_id: i for i in chunk}
        for tag_id, item in reasons.items():
            raw = by_tag.get(tag_id)
            if raw is None:
                # A tag the model silently dropped is recorded as uncertain
                # rather than quietly disappearing from the audit trail.
                judgments.append(
                    _record(
                        company,
                        tag_id,
                        item,
                        "uncertain",
                        0.0,
                        "model returned no judgment for this tag",
                        [],
                        "omitted by model",
                        config,
                        run_id,
                        model_digest,
                        PIPELINE_VERSION,
                    )
                )
                continue
            decision = str(raw.get("decision", "uncertain"))
            confidence = float(raw.get("confidence", 0.0))
            confidence = min(1.0, max(0.0, confidence))
            evidence = _verify_evidence(raw.get("evidence") or [], documents)
            # A positive claim with no verifiable evidence is downgraded: the
            # release gate requires provenance for every published assignment.
            note = raw.get("notes") or None
            if decision == "yes" and not evidence:
                decision, note = "uncertain", (note or "no verifiable evidence span")
            judgments.append(
                _record(
                    company,
                    tag_id,
                    item,
                    decision,
                    confidence,
                    truncate(str(raw.get("rationale", "")), 300),
                    evidence,
                    note,
                    config,
                    run_id,
                    model_digest,
                    PIPELINE_VERSION,
                )
            )
    return judgments


def _record(
    company: CompanyNormalized,
    tag_id: str,
    item: ShortlistItem,
    decision: str,
    confidence: float,
    rationale: str,
    evidence: list[EvidenceSpan],
    notes: str | None,
    config: Config,
    run_id: str,
    model_digest: str | None,
    pipeline_version: str,
) -> CompanyTagJudgment:
    return CompanyTagJudgment(
        judgment_id=_judgment_id(company.company_id, tag_id, run_id),
        company_id=company.company_id,
        tag_id=tag_id,
        decision=decision,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
        evidence=evidence,
        notes=notes,
        shortlist_reason=item.reason,  # type: ignore[arg-type]
        retrieval_score=item.score,
        model=config.models.chat_model,
        model_digest=model_digest,
        prompt_version=PROMPT_VERSIONS["assign"],
        ontology_version=ONTOLOGY_VERSION,
        pipeline_version=pipeline_version,
        run_id=run_id,
        created_at=now(),
    )


def information_weight(prevalence: float, floor: float = 0.15) -> float:
    """Down-weight ubiquitous tags.

    ``log(1/p)`` normalised to (floor, 1]. A tag on every company carries no
    information and collapses to ``floor``; a rare tag approaches 1.
    """
    p = min(max(prevalence, 1e-6), 1.0)
    raw = math.log(1.0 / p) / math.log(1.0 / 1e-6)
    return floor + (1.0 - floor) * min(1.0, raw)


def calibrate(confidence: float, decision: str) -> float:
    """Map a raw model confidence onto a calibrated score.

    LLM confidences cluster near 1.0, so we apply a mild concave correction
    that preserves ordering while spreading the top of the range. Both the raw
    and calibrated values are stored, so a better calibration can be fitted
    from the review set later without re-running the model.
    """
    if decision != "yes":
        return 0.0
    c = min(max(confidence, 0.0), 1.0)
    return round(0.5 * c + 0.5 * (c**2), 6)


def build_features(
    judgments: list[CompanyTagJudgment],
    *,
    n_companies: int,
    min_confidence: float,
    weight_floor: float = 0.15,
    run_id: str = "",
) -> list[CompanyTagFeature]:
    """Turn positive judgments into the published sparse company x tag vector."""
    positives = [
        j
        for j in judgments
        if j.decision == "yes" and j.confidence >= min_confidence and j.evidence
    ]
    prevalence: dict[str, int] = defaultdict(int)
    for j in positives:
        prevalence[j.tag_id] += 1

    denom = max(1, n_companies)
    features: list[CompanyTagFeature] = []
    seen: set[tuple[str, str]] = set()
    for j in positives:
        key = (j.company_id, j.tag_id)
        if key in seen:
            continue
        seen.add(key)
        weight = information_weight(prevalence[j.tag_id] / denom, weight_floor)
        calibrated = calibrate(j.confidence, j.decision)
        features.append(
            CompanyTagFeature(
                company_id=j.company_id,
                tag_id=j.tag_id,
                present=1,
                raw_confidence=round(j.confidence, 6),
                calibrated_confidence=calibrated,
                information_weight=round(weight, 6),
                feature_value=round(calibrated * weight, 6),
                judgment_id=j.judgment_id,
                ontology_version=j.ontology_version,
                run_id=run_id or j.run_id,
            )
        )
    return features


async def assign_many(
    *,
    config: Config,
    client: OllamaClient,
    work: list[tuple[CompanyNormalized, list[SourceDocument], list[ShortlistItem]]],
    tags_by_id: dict[str, Tag],
    run_id: str,
    model_digest: str | None = None,
    on_result=None,
) -> list[CompanyTagJudgment]:
    """Run assignment over many companies with bounded concurrency.

    ``on_result`` is called with each company's judgments as they complete, so
    the caller can checkpoint after every company and resume after an interrupt.
    """
    sem = asyncio.Semaphore(config.tagging.concurrency)
    out: list[CompanyTagJudgment] = []
    lock = asyncio.Lock()

    async def one(company, docs, shortlist) -> None:
        async with sem:
            judgments = await assign_tags(
                config=config,
                client=client,
                company=company,
                documents=docs,
                tags_by_id=tags_by_id,
                shortlist=shortlist,
                run_id=run_id,
                model_digest=model_digest,
            )
        async with lock:
            out.extend(judgments)
            if on_result is not None:
                on_result(company, judgments)

    await asyncio.gather(*(one(c, d, s) for c, d, s in work))
    return out
