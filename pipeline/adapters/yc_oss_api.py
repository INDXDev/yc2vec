"""Primary structured source: the open-source ``yc-oss/api`` project.

yc-oss republishes YC's public Algolia index as static JSON under a permissive
licence and refreshes it regularly. We pin the base URL, record the retrieval
timestamp and the upstream ``last_updated`` stamp, and preserve the payload
verbatim so that every downstream field stays traceable.
"""

from __future__ import annotations

from typing import Any

import httpx

from pipeline.adapters.base import AdapterResult
from pipeline.config import Config
from pipeline.models import CompanyRaw, SourceTaxonomyTerm
from pipeline.util import log, now, slugify, stable_hash

LOG = log(__name__)

#: yc-oss classification collections we ingest as first-class source taxonomy.
TAXONOMY_COLLECTIONS: tuple[tuple[str, str], ...] = (
    ("industries", "industry"),
    ("tags", "tag"),
    ("batches", "batch"),
)


class YcOssApiAdapter:
    name = "yc_oss_api"
    enabled_by_default = True

    def __init__(self, config: Config) -> None:
        self.config = config
        self.base = config.source_base_url.rstrip("/")

    async def fetch(self, limit: int | None = None) -> AdapterResult:
        if self.config.profile == "fixture":
            # The fixture profile is offline by construction: a fresh clone can
            # run the whole pipeline from the committed sample with no network.
            meta, companies = self._load_fixture()
        else:
            headers = {"User-Agent": self.config.user_agent, "Accept": "application/json"}
            async with httpx.AsyncClient(
                timeout=120.0, headers=headers, follow_redirects=True
            ) as client:
                meta = await self._get_json(client, f"{self.base}/meta.json")
                companies = await self._get_json(client, f"{self.base}/companies/all.json")

        retrieved_at = now()
        source_last_updated = meta.get("last_updated")
        if not isinstance(companies, list):
            return AdapterResult(self.name, errors=["companies/all.json was not a JSON array"])

        if limit is not None:
            companies = companies[:limit]

        records: list[CompanyRaw] = []
        errors: list[str] = []
        for payload in companies:
            if not isinstance(payload, dict) or "id" not in payload or "name" not in payload:
                errors.append(f"skipped malformed record: {str(payload)[:120]}")
                continue
            records.append(
                CompanyRaw(
                    company_id=f"ycoss:{payload['id']}",
                    source=self.name,
                    source_url=f"{self.base}/companies/all.json",
                    retrieved_at=retrieved_at,
                    source_last_updated=source_last_updated,
                    payload=payload,
                    content_hash=stable_hash(payload),
                )
            )

        terms = self._taxonomy_terms(meta, retrieved_at)
        LOG.info("yc_oss_api: %d companies, %d taxonomy terms", len(records), len(terms))
        return AdapterResult(
            name=self.name,
            records=records,
            meta={
                "source_last_updated": source_last_updated,
                "retrieved_at": retrieved_at,
                "source_url": f"{self.base}/companies/all.json",
                "taxonomy_terms": terms,
                "company_count": len(records),
            },
            errors=errors,
        )

    def _taxonomy_terms(self, meta: dict[str, Any], retrieved_at: Any) -> list[SourceTaxonomyTerm]:
        """Ingest yc-oss classifications with their exact original names and paths."""
        terms: list[SourceTaxonomyTerm] = []
        for collection, kind in TAXONOMY_COLLECTIONS:
            for slug, info in (meta.get(collection) or {}).items():
                if not isinstance(info, dict):
                    continue
                terms.append(
                    SourceTaxonomyTerm(
                        term_id=f"{kind}:{slug}",
                        kind=kind,  # type: ignore[arg-type]
                        slug=slug,
                        name=info.get("name") or slug,
                        company_count=int(info.get("count") or 0),
                        source_path=info.get("api"),
                        retrieved_at=retrieved_at,
                    )
                )
        return terms

    @staticmethod
    def derived_terms(
        payloads: list[dict[str, Any]], retrieved_at: Any
    ) -> list[SourceTaxonomyTerm]:
        """Terms that only exist inside company records (subindustry, region, stage, status).

        ``subindustry`` arrives as ``"Parent -> Child"``; we keep the exact string
        as the display name and record the parent relationship rather than
        inventing a new hierarchy.
        """
        seen: dict[str, SourceTaxonomyTerm] = {}
        counts: dict[str, int] = {}
        for p in payloads:
            fields: list[tuple[str, str, str | None]] = []
            if p.get("subindustry"):
                parent = str(p["subindustry"]).split("->")[0].strip()
                fields.append(("subindustry", str(p["subindustry"]), f"industry:{slugify(parent)}"))
            for r in p.get("regions") or []:
                fields.append(("region", str(r), None))
            if p.get("stage"):
                fields.append(("stage", str(p["stage"]), None))
            if p.get("status"):
                fields.append(("status", str(p["status"]), None))
            for kind, name, parent_id in fields:
                slug = slugify(name)
                term_id = f"{kind}:{slug}"
                counts[term_id] = counts.get(term_id, 0) + 1
                if term_id not in seen:
                    seen[term_id] = SourceTaxonomyTerm(
                        term_id=term_id,
                        kind=kind,  # type: ignore[arg-type]
                        slug=slug,
                        name=name,
                        parent_term_id=parent_id,
                        retrieved_at=retrieved_at,
                    )
        for term_id, term in seen.items():
            term.company_count = counts.get(term_id, 0)
        return sorted(seen.values(), key=lambda t: t.term_id)

    def _load_fixture(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "companies_sample.json"
        companies = json.loads(path.read_text())
        # Reconstruct the taxonomy collections meta.json would have carried, so
        # the fixture exercises the same code path as the live source.
        meta: dict[str, Any] = {
            "last_updated": "fixture",
            "industries": {},
            "tags": {},
            "batches": {},
        }
        for c in companies:
            for collection, values in (
                ("industries", [c.get("industry")] + (c.get("industries") or [])),
                ("tags", c.get("tags") or []),
                ("batches", [c.get("batch")]),
            ):
                for value in values:
                    if not value:
                        continue
                    slug = slugify(value)
                    entry = meta[collection].setdefault(
                        slug, {"name": value, "count": 0, "api": None}
                    )
                    entry["count"] += 1
        return meta, companies

    @staticmethod
    async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
        LOG.info("GET %s", url)
        r = await client.get(url)
        r.raise_for_status()
        return r.json()
