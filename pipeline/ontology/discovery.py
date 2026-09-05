"""Stage 1: open-ended candidate discovery.

We show the model small, *diverse* batches of companies and ask what reusable
semantic attributes distinguish them. Diversity matters: batching by similarity
produces narrow, redundant tags, so batches are built by striding across a
deterministic ordering that interleaves industries and batches.

Discovery is deliberately generous. Nothing proposed here becomes an active tag
without passing normalisation, merge review and an activation threshold.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from pipeline.config import Config
from pipeline.models import CompanyNormalized, TagCandidate
from pipeline.ollama import OllamaClient
from pipeline.ontology.registry import OntologyRegistry
from pipeline.prompts import (
    DISCOVERY_SCHEMA,
    DISCOVERY_SYSTEM,
    PROMPT_VERSIONS,
    discovery_prompt,
)
from pipeline.util import log, normalize_name, now, slugify, truncate

LOG = log(__name__)


def diverse_batches(
    companies: list[CompanyNormalized], batch_size: int
) -> list[list[CompanyNormalized]]:
    """Group companies so each batch spans unrelated industries and years.

    Sorting by (industry, batch_year) and then striding with a step equal to the
    number of batches gives every batch at most one company per industry run,
    which is what makes the discovery prompt produce cross-cutting attributes
    instead of restating one vertical.
    """
    if not companies:
        return []
    ordered = sorted(companies, key=lambda c: (c.industry or "zz", c.batch_year or 0, c.company_id))
    n_batches = max(1, len(ordered) // max(1, batch_size))
    buckets: list[list[CompanyNormalized]] = [[] for _ in range(n_batches)]
    for i, company in enumerate(ordered):
        buckets[i % n_batches].append(company)
    return [b for b in buckets if b]


def _company_text(c: CompanyNormalized) -> str:
    bits = [c.one_liner or "", c.long_description or ""]
    meta = []
    if c.industry:
        meta.append(f"industry: {c.industry}")
    if c.source_tags:
        meta.append("yc tags: " + ", ".join(c.source_tags[:6]))
    return truncate(" ".join(b for b in bits if b) + (" | " + "; ".join(meta) if meta else ""), 900)


async def discover_candidates(
    *,
    config: Config,
    client: OllamaClient,
    registry: OntologyRegistry,
    companies: list[CompanyNormalized],
    run_id: str,
    max_batches: int | None = None,
    concurrency: int = 4,
) -> list[TagCandidate]:
    """Run discovery over diverse batches and record normalised candidates."""
    facets = config.ontology.facets
    batches = diverse_batches(companies, config.ontology.discovery_batch_size)
    if max_batches is not None:
        batches = batches[:max_batches]
    LOG.info(
        "discovery: %d batches of ~%d companies", len(batches), config.ontology.discovery_batch_size
    )

    sem = asyncio.Semaphore(concurrency)
    collected: list[TagCandidate] = []
    lock = asyncio.Lock()

    async def one(batch: list[CompanyNormalized], idx: int) -> None:
        payload = [
            {"company_id": c.company_id, "name": c.name, "text": _company_text(c)} for c in batch
        ]
        prompt = discovery_prompt(payload, facets, config.ontology.max_candidates_per_batch)
        async with sem:
            try:
                result: dict[str, Any] = await client.generate_json(
                    system=DISCOVERY_SYSTEM,
                    prompt=prompt,
                    schema=DISCOVERY_SCHEMA,
                    prompt_version=PROMPT_VERSIONS["discovery"],
                    num_predict=2400,
                    cache_namespace="discovery",
                )
            except Exception as exc:  # a bad batch must not kill the stage
                LOG.warning("discovery batch %d failed: %s", idx, exc)
                return

        valid_ids = {c.company_id for c in batch}
        out: list[TagCandidate] = []
        for item in result.get("candidates", []):
            candidate = _to_candidate(item, facets, valid_ids, run_id, config.models.chat_model)
            if candidate is not None:
                out.append(candidate)
        async with lock:
            for candidate in out:
                collected.append(registry.add_candidate(candidate))
        LOG.debug("discovery batch %d -> %d candidates", idx, len(out))

    await asyncio.gather(*(one(b, i) for i, b in enumerate(batches)))
    LOG.info("discovery: %d candidate proposals recorded", len(collected))
    return collected


def _to_candidate(
    item: dict[str, Any],
    facets: tuple[str, ...],
    valid_ids: set[str],
    run_id: str,
    model: str,
) -> TagCandidate | None:
    """Normalise one raw proposal, or drop it.

    The model occasionally invents a facet or cites a company that was not in
    the batch. Rather than trusting either, we snap the facet to the controlled
    list and intersect the support ids with the batch we actually showed it.
    """
    name = _display_name(str(item.get("name", "")))
    definition = " ".join(str(item.get("definition", "")).split())
    if len(name) < 2 or len(definition) < 20:
        return None
    raw_facet = str(item.get("facet", "")).strip().lower().replace(" ", "_").replace("-", "_")
    facet: str | None = raw_facet if raw_facet in facets else _closest_facet(raw_facet, facets)
    if facet is None:
        return None
    support = sorted(set(item.get("supporting_company_ids") or []) & valid_ids)
    if not support:
        return None
    return TagCandidate(
        candidate_id=f"{facet}:{slugify(name)}",
        proposed_name=name,
        normalized_name=normalize_name(name),
        facet=facet,
        definition=definition,
        positive_examples=[str(x) for x in (item.get("positive_examples") or [])][:4],
        negative_examples=[str(x) for x in (item.get("negative_examples") or [])][:4],
        support_company_ids=support,
        discovery_run_id=run_id,
        model=model,
        prompt_version=PROMPT_VERSIONS["discovery"],
        created_at=now(),
    )


#: Domain acronyms the model routinely emits in lower case. The value is the
#: preferred rendering, which is not always a plain upper-casing.
_ACRONYMS: dict[str, str] = {
    "ai": "AI",
    "api": "API",
    "apis": "APIs",
    "saas": "SaaS",
    "paas": "PaaS",
    "iaas": "IaaS",
    "iot": "IoT",
    "ml": "ML",
    "llm": "LLM",
    "llms": "LLMs",
    "nlp": "NLP",
    "ocr": "OCR",
    "crm": "CRM",
    "erp": "ERP",
    "hr": "HR",
    "seo": "SEO",
    "sdk": "SDK",
    "ui": "UI",
    "ux": "UX",
    "kyc": "KYC",
    "aml": "AML",
    "ehr": "EHR",
    "emr": "EMR",
    "ev": "EV",
    "evs": "EVs",
    "gpu": "GPU",
    "cpu": "CPU",
    "sql": "SQL",
    "pos": "POS",
    "ci": "CI",
    "cd": "CD",
    "devops": "DevOps",
    "fintech": "Fintech",
    "3d": "3D",
}

_LETTER_DIGIT_LETTER = re.compile(r"[a-z]\d[a-z]")


def _display_name(raw: str) -> str:
    """Turn a model-emitted identifier into a readable canonical name.

    Models drift between ``AI Agents``, ``ai_agents`` and ``AI-Agents``. The
    display name is normalised here; the tag *id* is minted separately and never
    changes afterwards, so this only affects presentation.
    """
    text = " ".join(raw.replace("_", " ").replace("-", " ").split())
    if not text:
        return ""
    words = []
    for w in text.split():
        # Preserve deliberate acronyms rather than title-casing them into
        # nonsense ("B2b Saa S"). Three rules cover what the model emits:
        # already-uppercase tokens, the well-known domain acronyms, and the
        # letter-digit-letter shorthands (b2b, b2c, p2p, o2o).
        lower = w.lower()
        if w.isupper() or (len(w) <= 4 and sum(c.isupper() for c in w) >= 2):
            words.append(w)
        elif lower in _ACRONYMS or _LETTER_DIGIT_LETTER.fullmatch(lower):
            words.append(_ACRONYMS.get(lower, lower.upper()))
        else:
            words.append(w[0].upper() + w[1:] if w[0].islower() else w)
    return " ".join(words)


def _closest_facet(value: str, facets: tuple[str, ...]) -> str | None:
    """Snap a near-miss facet name ('business model') onto the controlled list."""
    if not value:
        return None
    for f in facets:
        if value == f or value.startswith(f) or f.startswith(value):
            return f
    tokens = set(value.split("_"))
    best, best_overlap = None, 0
    for f in facets:
        overlap = len(tokens & set(f.split("_")))
        if overlap > best_overlap:
            best, best_overlap = f, overlap
    return best
