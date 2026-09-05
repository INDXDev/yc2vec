"""The fetch stage: source records, taxonomy terms, normalisation, enrichment.

Everything downstream reads the artifacts this stage writes, so it is the only
place that talks to the network for structured data. It is incremental: a
company whose source ``content_hash`` is unchanged keeps its existing
normalised record and its cached website text.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.adapters import CompanyWebsiteAdapter, YcOssApiAdapter
from pipeline.config import Config
from pipeline.models import CompanyNormalized, CompanyRaw, SourceTaxonomyTerm, WebSource
from pipeline.normalize.companies import normalize_company
from pipeline.store import Store
from pipeline.util import log, read_jsonl, write_jsonl

LOG = log(__name__)


async def fetch_stage(
    config: Config,
    store: Store,
    *,
    limit: int | None = None,
    changed_since_hashes: dict[str, str] | None = None,
) -> dict[str, int]:
    adapter = YcOssApiAdapter(config)
    result = await adapter.fetch(limit=limit)
    if result.errors:
        LOG.warning("source adapter reported %d malformed records", len(result.errors))

    raws: list[CompanyRaw] = result.records
    retrieved_at = result.meta["retrieved_at"]

    terms: list[SourceTaxonomyTerm] = list(result.meta["taxonomy_terms"])
    terms.extend(YcOssApiAdapter.derived_terms([r.payload for r in raws], retrieved_at))
    # Deduplicate, preferring the meta.json record which carries the source path.
    by_id: dict[str, SourceTaxonomyTerm] = {}
    for t in terms:
        existing = by_id.get(t.term_id)
        if existing is None or (not existing.source_path and t.source_path):
            by_id[t.term_id] = t
    terms = sorted(by_id.values(), key=lambda t: t.term_id)

    write_jsonl(store.path("raw", "companies_raw.jsonl"), raws)
    write_jsonl(store.path("raw", "source_taxonomy_terms.jsonl"), terms)

    previous = {r["company_id"]: r for r in read_jsonl(store.path("normalized", "companies.jsonl"))}
    changed: list[str] = []
    normalized: list[CompanyNormalized] = []
    for raw in raws:
        prior = previous.get(raw.company_id)
        prior_hash = (changed_since_hashes or {}).get(raw.company_id)
        if prior is not None and prior_hash == raw.content_hash:
            normalized.append(CompanyNormalized(**prior))
            continue
        record = normalize_company(raw)
        if prior is None or prior.get("content_hash") != record.content_hash:
            changed.append(record.company_id)
        normalized.append(record)

    write_jsonl(store.path("normalized", "companies.jsonl"), normalized)
    store.cache_put("source", "raw_hashes", {r.company_id: r.content_hash for r in raws})

    counts = {
        "companies": len(normalized),
        "taxonomy_terms": len(terms),
        "changed": len(changed),
    }

    if config.crawl.enabled:
        counts.update(await _enrich(config, store, normalized))

    LOG.info("fetch: %s", counts)
    return counts


async def _enrich(
    config: Config, store: Store, companies: list[CompanyNormalized]
) -> dict[str, int]:
    """Optional website enrichment. Failures never discard a previous success."""
    targets = [(c.company_id, c.website) for c in companies if c.website]
    if not targets:
        return {"web_sources": 0}
    adapter = CompanyWebsiteAdapter(config, store)
    result = await adapter.fetch(targets)

    path = store.path("raw", "web_sources.jsonl")
    previous: dict[str, dict] = {}
    for row in read_jsonl(path):
        if row.get("ok"):
            previous[f"{row['company_id']}|{row['url']}"] = row

    merged: dict[str, WebSource] = {}
    for src in result.records:
        key = f"{src.company_id}|{src.url}"
        if not src.ok and key in previous:
            merged[key] = WebSource(**previous[key])
        else:
            merged[key] = src
    write_jsonl(path, sorted(merged.values(), key=lambda s: (s.company_id, s.url)))

    texts = result.meta.get("texts") or {}
    combined = {
        cid: "\n\n".join(p.text for p in pages)[:6000] for cid, pages in texts.items() if pages
    }
    store.cache_put("web", "company_texts", combined)
    return {"web_sources": len(merged), "web_texts": len(combined)}


def load_website_texts(store: Store) -> dict[str, str]:
    return store.cache_get("web", "company_texts") or {}


def load_normalized(store: Store) -> list[CompanyNormalized]:
    return [CompanyNormalized(**r) for r in read_jsonl(store.path("normalized", "companies.jsonl"))]


def load_terms(store: Store) -> list[SourceTaxonomyTerm]:
    return [
        SourceTaxonomyTerm(**r)
        for r in read_jsonl(store.path("raw", "source_taxonomy_terms.jsonl"))
    ]


def source_meta(store: Store) -> dict[str, str]:
    raw_path: Path = store.path("raw", "companies_raw.jsonl")
    for row in read_jsonl(raw_path):
        return {
            "source_url": row.get("source_url", ""),
            "retrieved_at": row.get("retrieved_at", ""),
            "source_last_updated": row.get("source_last_updated") or "",
        }
    return {}
