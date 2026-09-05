"""Turn raw source records into typed, filterable company records.

Two things happen here and nowhere else:

* **Field normalisation.** Types are coerced, batches are parsed into season and
  year, and source classifications are linked to ``source_taxonomy_terms`` by
  id. Missing stays missing -- we never invent a value, and never let the LLM
  fill one in.
* **The metadata document.** A deterministic natural-language serialisation of
  the publishable normalised fields. It is embedded as its own vector so that
  "similar metadata" is a first-class, inspectable similarity mode, distinct
  from description text and from inferred semantic tags.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pipeline.models import CompanyNormalized, CompanyRaw, SourceDocument
from pipeline.util import now, sha256_text, slugify, stable_hash
from pipeline.versions import EXTRACTION_VERSION, METADATA_TEMPLATE_VERSION, NORMALIZE_VERSION

_BATCH_RE = re.compile(r"^(Winter|Summer|Spring|Fall|IK12|Unspecified)\s*(\d{4})?$", re.IGNORECASE)
_SEASONS = {"winter": "Winter", "summer": "Summer", "spring": "Spring", "fall": "Fall"}


def parse_batch(batch: str | None) -> tuple[str, int | None, str | None]:
    """``"Winter 2012"`` -> ``("Winter", 2012, "winter-2012")``.

    Unparseable or historical batch labels (``IK12``, ``Unspecified``) keep
    their exact source string as the slug and report no season/year rather than
    being coerced into a guess.
    """
    if not batch:
        return "Unspecified", None, None
    m = _BATCH_RE.match(batch.strip())
    if not m:
        return "Unspecified", None, slugify(batch)
    season = _SEASONS.get((m.group(1) or "").lower(), "Unspecified")
    year = int(m.group(2)) if m.group(2) else None
    return season, year, slugify(batch)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _int(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        cleaned = _clean(item)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def build_metadata_document(c: CompanyNormalized) -> str:
    """Deterministic NL serialisation of publishable metadata.

    Only fields that are present appear, in a fixed order, so the same company
    always yields byte-identical text and therefore a cache-stable embedding.
    Descriptions are deliberately excluded: they belong to the description
    embedding, and mixing them here would make "metadata similarity"
    meaningless.
    """
    parts: list[str] = [f"{c.name} is a Y Combinator company."]
    if c.batch:
        if c.batch_year:
            parts.append(f"It participated in the {c.batch} batch, in {c.batch_year}.")
        else:
            parts.append(f"It participated in the {c.batch} batch.")
    if c.status:
        parts.append(f"Its status is {c.status.lower()}.")
    if c.stage:
        parts.append(f"Its stage is {c.stage.lower()}.")
    if c.industry:
        parts.append(f"Y Combinator lists its industry as {c.industry}.")
    if c.subindustry and c.subindustry != c.industry:
        parts.append(f"Its sub-industry is {c.subindustry.replace('->', 'within').strip()}.")
    extra = [i for i in c.industries if i not in {c.industry}]
    if extra:
        parts.append(f"Additional listed industries: {', '.join(extra)}.")
    if c.source_tags:
        parts.append(f"Y Combinator source tags: {', '.join(c.source_tags)}.")
    if c.all_locations:
        parts.append(f"It is located in {c.all_locations}.")
    if c.regions:
        parts.append(f"Listed regions: {', '.join(c.regions)}.")
    if c.team_size is not None:
        parts.append(f"Its team size is {c.team_size}.")
    if c.nonprofit:
        parts.append("It is a nonprofit.")
    if c.top_company:
        parts.append("Y Combinator lists it as a top company.")
    if c.is_hiring:
        parts.append("It is currently hiring.")
    if c.launched_at:
        parts.append(f"It launched in {c.launched_at.year}.")
    return " ".join(parts)


def normalize_company(raw: CompanyRaw) -> CompanyNormalized:
    p = raw.payload
    season, year, batch_slug = parse_batch(_clean(p.get("batch")))

    launched: datetime | None = None
    ts = p.get("launched_at")
    if isinstance(ts, int | float) and ts > 0:
        try:
            launched = datetime.fromtimestamp(float(ts), tz=UTC)
        except (OverflowError, OSError, ValueError):
            launched = None

    industry = _clean(p.get("industry"))
    subindustry = _clean(p.get("subindustry"))
    industries = _str_list(p.get("industries"))
    source_tags = _str_list(p.get("tags"))
    regions = _str_list(p.get("regions"))
    status = _clean(p.get("status"))
    stage = _clean(p.get("stage"))

    term_ids: list[str] = []
    for kind, values in (
        ("industry", [industry] if industry else []),
        ("industry", industries),
        ("subindustry", [subindustry] if subindustry else []),
        ("tag", source_tags),
        ("batch", [_clean(p.get("batch"))] if p.get("batch") else []),
        ("region", regions),
        ("stage", [stage] if stage else []),
        ("status", [status] if status else []),
    ):
        for v in values:
            if v:
                tid = f"{kind}:{slugify(v)}"
                if tid not in term_ids:
                    term_ids.append(tid)

    company = CompanyNormalized(
        company_id=raw.company_id,
        yc_id=int(p["id"]),
        slug=_clean(p.get("slug")) or slugify(str(p.get("name", ""))),
        name=_clean(p.get("name")) or "Unknown",
        former_names=_str_list(p.get("former_names")),
        one_liner=_clean(p.get("one_liner")),
        long_description=_clean(p.get("long_description")),
        website=_clean(p.get("website")),
        yc_url=_clean(p.get("url")) or f"https://www.ycombinator.com/companies/{p.get('slug', '')}",
        logo_url=_clean(p.get("small_logo_thumb_url")),
        batch=_clean(p.get("batch")),
        batch_slug=batch_slug,
        batch_season=season,  # type: ignore[arg-type]
        batch_year=year,
        status=status,
        stage=stage,
        team_size=_int(p.get("team_size")),
        is_hiring=bool(p.get("isHiring")),
        nonprofit=bool(p.get("nonprofit")),
        top_company=bool(p.get("top_company")),
        industry=industry,
        subindustry=subindustry,
        industries=industries,
        source_tags=source_tags,
        regions=regions,
        all_locations=_clean(p.get("all_locations")),
        launched_at=launched,
        source_taxonomy_term_ids=term_ids,
        metadata_document="",
        metadata_template_version=METADATA_TEMPLATE_VERSION,
        normalize_version=NORMALIZE_VERSION,
        content_hash="",
        updated_at=now(),
    )
    company.metadata_document = build_metadata_document(company)
    payload = company.model_dump(mode="json", exclude={"content_hash", "updated_at"})
    company.content_hash = stable_hash(payload)
    return company


def normalize_companies(raws: list[CompanyRaw]) -> list[CompanyNormalized]:
    return [normalize_company(r) for r in raws]


def company_documents(
    company: CompanyNormalized, website_text: str | None = None
) -> list[SourceDocument]:
    """The evidence set a tag judgment may quote from.

    ``document_id`` is stable and appears in every judgment, so a rationale can
    always be traced to the exact text that produced it.
    """
    docs: list[SourceDocument] = []

    def add(kind: str, text: str | None, source_url: str | None) -> None:
        if not text or not text.strip():
            return
        docs.append(
            SourceDocument(
                document_id=f"{company.company_id}#{kind}",
                company_id=company.company_id,
                kind=kind,  # type: ignore[arg-type]
                text=text.strip(),
                char_count=len(text.strip()),
                source_url=source_url,
                content_hash=sha256_text(text.strip()),
                extraction_version=EXTRACTION_VERSION,
                created_at=company.updated_at,
            )
        )

    add("yc_one_liner", company.one_liner, company.yc_url)
    add("yc_long_description", company.long_description, company.yc_url)
    add("metadata_document", company.metadata_document, None)
    add("website_main_text", website_text, company.website)
    return docs


def description_document(company: CompanyNormalized, website_text: str | None = None) -> str:
    """Canonical text embedded as ``description_embedding``."""
    parts = [company.name]
    if company.one_liner:
        parts.append(company.one_liner)
    if company.long_description:
        parts.append(company.long_description)
    if website_text:
        parts.append(website_text[:2000])
    return "\n".join(parts)
