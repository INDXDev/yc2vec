"""Generate the static artifacts the browser loads.

Layout (all under ``data/public/v1/``)::

    manifest.json          boot manifest: versions, counts, shard layout, checksums
    points.json            parallel arrays for the scatterplot -- loaded first
    companies.json         compact per-company index for lists, filters, search
    tags.json              the active ontology plus prevalence and co-occurrence
    taxonomy.json          exact YC source terms and their mappings to our tags
    clusters.json          algorithmic cluster labels
    search/docs.json       field-projected search documents
    detail/<shard>.json    full company detail: evidence, rationale, neighbours

Two rules drive the format. First, the map must paint before the heavy data
arrives, so coordinates live in ``points.json`` as parallel typed arrays rather
than as objects. Second, nothing here may require a backend: every filter,
search and similarity lookup is answered from these files.

Keys are shortened only where the saving is large (the point arrays and the
per-company index); the mapping is documented in ``manifest.json`` under
``key_map`` so the artifacts stay self-describing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.models import (
    Cluster,
    CompanyNormalized,
    CompanyTagFeature,
    CompanyTagJudgment,
    Neighbor,
    ReleaseManifest,
    SourceTaxonomyTagMapping,
    SourceTaxonomyTerm,
    Tag,
    UmapPoint,
)
from pipeline.util import log, sha256_file, truncate, write_json
from pipeline.versions import PUBLIC_ARTIFACT_VERSION, SCHEMA_VERSION

LOG = log(__name__)

#: Number of per-company detail shards. At the full corpus this puts roughly a
#: hundred companies in each shard, so opening one company fetches about 1/64th
#: of the detail data instead of all of it.
DETAIL_SHARDS = 64

#: Neighbours published per similarity space. The pipeline computes and exports
#: more (see EmbeddingConfig.top_k_neighbors); the browser only ever shows the
#: head of the list, and shipping the tail would dominate the detail payload.
PUBLISHED_NEIGHBORS_PER_SPACE = 12

#: Short transport keys for the company index, documented in the manifest.
COMPANY_KEY_MAP = {
    "i": "company_id",
    "n": "name",
    "o": "one_liner",
    "b": "batch",
    "y": "batch_year",
    "s": "status",
    "g": "stage",
    "d": "industry",
    "u": "subindustry",
    "r": "regions",
    "l": "all_locations",
    "t": "team_size",
    "w": "website",
    "c": "yc_url",
    "m": "logo_url",
    "k": "source_tags",
    "p": "top_company",
    "h": "is_hiring",
    "z": "nonprofit",
    "T": "tag_ids",
    "S": "tag_scores",
}


def _company_index_row(c: CompanyNormalized, features: list[CompanyTagFeature]) -> dict[str, Any]:
    ranked = sorted(features, key=lambda f: -f.feature_value)
    row: dict[str, Any] = {
        "i": c.company_id,
        "n": c.name,
        "o": truncate(c.one_liner or "", 160),
        "b": c.batch,
        "y": c.batch_year,
        "s": c.status,
        "g": c.stage,
        "d": c.industry,
        "u": c.subindustry,
        "r": c.regions,
        "l": c.all_locations,
        "t": c.team_size,
        "w": c.website,
        "c": c.yc_url,
        "m": c.logo_url,
        "k": c.source_tags,
        "T": [f.tag_id for f in ranked],
        "S": [round(f.feature_value, 4) for f in ranked],
    }
    if c.top_company:
        row["p"] = 1
    if c.is_hiring:
        row["h"] = 1
    if c.nonprofit:
        row["z"] = 1
    # Drop nulls and empties: across thousands of rows this is a large saving
    # and the frontend already treats a missing key as "unknown".
    return {k: v for k, v in row.items() if v not in (None, "", [], 0) or k in ("i", "n")}


def shard_for(company_id: str) -> int:
    """Stable shard assignment. Must match the frontend's implementation."""
    h = 0
    for ch in company_id:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h % DETAIL_SHARDS


def publish_browser_artifacts(
    out_dir: Path,
    *,
    companies: list[CompanyNormalized],
    tags: list[Tag],
    features: list[CompanyTagFeature],
    judgments: list[CompanyTagJudgment],
    neighbors: list[Neighbor],
    points: list[UmapPoint],
    clusters: list[Cluster],
    terms: list[SourceTaxonomyTerm],
    mappings: list[SourceTaxonomyTagMapping],
    manifest: ReleaseManifest,
) -> dict[str, int]:
    root = out_dir / f"v{PUBLIC_ARTIFACT_VERSION}"
    root.mkdir(parents=True, exist_ok=True)

    by_company_features: dict[str, list[CompanyTagFeature]] = defaultdict(list)
    for f in features:
        by_company_features[f.company_id].append(f)
    tags_by_id = {t.tag_id: t for t in tags}
    points_by_company = {p.company_id: p for p in points}

    # Only publish companies that made it all the way through the pipeline, so
    # the map, the index and the detail shards can never disagree.
    published = [c for c in companies if c.company_id in points_by_company]
    published.sort(key=lambda c: c.company_id)
    order = {c.company_id: i for i, c in enumerate(published)}

    counts: dict[str, int] = {}

    # -- points.json: parallel arrays, smallest possible first paint --------
    xs, ys, cl, yr = [], [], [], []
    for c in published:
        p = points_by_company[c.company_id]
        xs.append(round(p.x, 3))
        ys.append(round(p.y, 3))
        cl.append(p.cluster_id)
        yr.append(c.batch_year or 0)
    write_json(
        root / "points.json",
        {
            "version": manifest.projection_version,
            "embedding_space_version": manifest.embedding_space_version,
            "count": len(published),
            "ids": [c.company_id for c in published],
            "x": xs,
            "y": ys,
            "cluster": cl,
            "year": yr,
            "note": (
                "UMAP is a lossy 2D projection. Distance on this chart does not "
                "faithfully preserve high-dimensional similarity; use the "
                "precomputed neighbour lists for similarity."
            ),
        },
    )
    counts["points"] = len(published)

    # -- companies.json: compact index ---------------------------------------
    write_json(
        root / "companies.json",
        {
            "key_map": COMPANY_KEY_MAP,
            "count": len(published),
            "rows": [
                _company_index_row(c, by_company_features.get(c.company_id, [])) for c in published
            ],
        },
    )
    counts["companies"] = len(published)

    # -- tags.json: ontology with prevalence and co-occurrence ---------------
    prevalence: dict[str, int] = defaultdict(int)
    company_tags: dict[str, list[str]] = defaultdict(list)
    for f in features:
        if f.company_id in order:
            prevalence[f.tag_id] += 1
            company_tags[f.company_id].append(f.tag_id)

    cooc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tag_ids in company_tags.values():
        for i, a in enumerate(tag_ids):
            for b in tag_ids[i + 1 :]:
                cooc[a][b] += 1
                cooc[b][a] += 1

    by_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cid, tag_ids in company_tags.items():
        company = published[order[cid]]
        if company.batch_year:
            for tag_id in tag_ids:
                by_year[tag_id][str(company.batch_year)] += 1

    # Publish the whole active ontology, not just the tags that happen to have
    # assignments yet. The ontology is a result in its own right, and hiding
    # the unassigned two-thirds would misrepresent both what was discovered and
    # how far assignment has got. Prevalence is published as-is -- including
    # zero -- and the UI is expected to say so rather than imply the tag is
    # unused in the world.
    published_tags = [t for t in tags if t.state == "active"]
    write_json(
        root / "tags.json",
        {
            "ontology_version": manifest.ontology_version,
            "count": len(published_tags),
            "facets": sorted({t.facet for t in published_tags}),
            "rows": [
                {
                    "tag_id": t.tag_id,
                    "name": t.canonical_name,
                    "facet": t.facet,
                    "definition": t.definition,
                    "aliases": t.aliases[:8],
                    "parents": t.parent_tag_ids,
                    "prevalence": prevalence.get(t.tag_id, 0),
                    "support": t.support_count,
                    "examples": [
                        published[order[cid]].name
                        for cid in t.source_company_ids[:4]
                        if cid in order
                    ],
                    "cooccurring": [
                        {"tag_id": k, "count": v}
                        for k, v in sorted(cooc[t.tag_id].items(), key=lambda kv: -kv[1])[:8]
                    ],
                    "by_year": dict(sorted(by_year[t.tag_id].items())),
                }
                for t in sorted(
                    # Most-assigned first, then best-supported. The second key
                    # matters early in a corpus's life, when most tags have no
                    # assignments yet and prevalence alone would order them
                    # alphabetically -- burying the tags discovery was most
                    # confident about.
                    published_tags,
                    key=lambda t: (-prevalence.get(t.tag_id, 0), -t.support_count, t.tag_id),
                )
            ],
        },
    )
    counts["tags"] = len(published_tags)

    # -- taxonomy.json: exact source terms, kept separate from our ontology --
    used_terms = {tid for c in published for tid in c.source_taxonomy_term_ids}
    write_json(
        root / "taxonomy.json",
        {
            "terms": [
                {
                    "term_id": t.term_id,
                    "kind": t.kind,
                    "name": t.name,
                    "parent": t.parent_term_id,
                    "count": t.company_count,
                }
                for t in sorted(terms, key=lambda t: t.term_id)
                if t.term_id in used_terms
            ],
            "mappings": [
                {
                    "term_id": m.term_id,
                    "tag_id": m.tag_id,
                    "relation": m.relation,
                    "similarity": round(m.similarity, 4),
                    "reviewed": m.reviewed,
                }
                for m in sorted(mappings, key=lambda m: (m.term_id, -m.similarity))
                if m.term_id in used_terms and m.tag_id in tags_by_id
            ],
        },
    )
    counts["taxonomy_terms"] = len(used_terms)

    # -- clusters.json --------------------------------------------------------
    write_json(
        root / "clusters.json",
        {
            "projection_version": manifest.projection_version,
            "disclaimer": "Clusters are algorithmic, not official YC categories.",
            "rows": [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "size": c.size,
                    "top_tag_ids": c.top_tag_ids,
                    "x": c.centroid_x,
                    "y": c.centroid_y,
                }
                for c in clusters
            ],
        },
    )
    counts["clusters"] = len(clusters)

    # -- search/docs.json: the static keyword-search payload -------------------
    # Built at publication time and queried entirely in the browser. Fields are
    # projected here so the client never has to reconstruct searchable text
    # from several artifacts.
    search_rows = []
    for c in published:
        tag_names = [
            tags_by_id[f.tag_id].canonical_name
            for f in sorted(
                by_company_features.get(c.company_id, []), key=lambda f: -f.feature_value
            )[:14]
            if f.tag_id in tags_by_id
        ]
        aliases = [
            a
            for f in by_company_features.get(c.company_id, [])[:8]
            if f.tag_id in tags_by_id
            for a in tags_by_id[f.tag_id].aliases[:2]
        ]
        search_rows.append(
            {
                "i": c.company_id,
                "n": c.name,
                "a": " ".join(c.former_names),
                "o": c.one_liner or "",
                "e": truncate(c.long_description or "", 420),
                "g": " ".join(tag_names),
                "x": " ".join(dict.fromkeys(aliases)),
                "d": " ".join(
                    filter(None, [c.industry, c.subindustry, *c.industries, *c.source_tags])
                ),
                "b": " ".join(
                    filter(
                        None, [c.batch, c.all_locations, *c.regions, c.status or "", c.stage or ""]
                    )
                ),
            }
        )
    write_json(root / "search" / "docs.json", {"count": len(search_rows), "rows": search_rows})
    counts["search_docs"] = len(search_rows)

    # -- detail shards ---------------------------------------------------------
    judgments_by_company: dict[str, list[CompanyTagJudgment]] = defaultdict(list)
    for j in judgments:
        judgments_by_company[j.company_id].append(j)
    neighbors_by_company: dict[str, list[Neighbor]] = defaultdict(list)
    for nb in neighbors:
        neighbors_by_company[nb.company_id].append(nb)

    tag_sets: dict[str, set[str]] = {
        cid: {f.tag_id for f in feats} for cid, feats in by_company_features.items()
    }
    # Metadata facts are compared for every (company, neighbour) pair. At full
    # corpus size that is hundreds of thousands of comparisons, so build each
    # company's set once rather than per pair.
    fact_sets: dict[str, set[str]] = {c.company_id: _metadata_facts(c) for c in published}
    shards: dict[int, dict[str, Any]] = defaultdict(dict)
    for c in published:
        shards[shard_for(c.company_id)][c.company_id] = _detail_record(
            c,
            by_company_features.get(c.company_id, []),
            judgments_by_company.get(c.company_id, []),
            neighbors_by_company.get(c.company_id, []),
            tags_by_id,
            order,
            published,
            points_by_company,
            tag_sets,
            fact_sets,
        )
    for shard_id in range(DETAIL_SHARDS):
        write_json(root / "detail" / f"{shard_id}.json", shards.get(shard_id, {}))
    counts["detail_shards"] = DETAIL_SHARDS

    # -- manifest --------------------------------------------------------------
    checksums: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        checksums[str(path.relative_to(root))] = sha256_file(path)[:16]

    manifest.counts = {**manifest.counts, **counts}
    manifest.checksums = checksums
    manifest.schema_version = SCHEMA_VERSION
    manifest.public_artifact_version = PUBLIC_ARTIFACT_VERSION
    payload = manifest.model_dump(mode="json")
    payload["detail_shards"] = DETAIL_SHARDS
    payload["key_map"] = COMPANY_KEY_MAP
    payload["files"] = sorted(checksums)
    write_json(root / "manifest.json", payload, pretty=True)

    # A stable pointer so the app can discover the current release version.
    write_json(
        out_dir / "index.json",
        {
            "current": f"v{PUBLIC_ARTIFACT_VERSION}",
            "dataset_version": manifest.dataset_version,
            "generated_at": manifest.generated_at
            if isinstance(manifest.generated_at, str)
            else manifest.generated_at.isoformat(),
        },
    )
    LOG.info("published browser artifacts to %s (%s)", root, counts)
    return counts


def _detail_record(
    c: CompanyNormalized,
    features: list[CompanyTagFeature],
    judgments: list[CompanyTagJudgment],
    neighbors: list[Neighbor],
    tags_by_id: dict[str, Tag],
    order: dict[str, int],
    published: list[CompanyNormalized],
    points_by_company: dict[str, UmapPoint],
    neighbor_tags: dict[str, set[str]],
    neighbor_facts: dict[str, set[str]],
) -> dict[str, Any]:
    by_tag = {j.tag_id: j for j in judgments if j.decision == "yes"}
    tag_rows = []
    for f in sorted(features, key=lambda f: -f.feature_value):
        tag = tags_by_id.get(f.tag_id)
        if tag is None:
            continue
        j = by_tag.get(f.tag_id)
        tag_rows.append(
            {
                "tag_id": f.tag_id,
                "name": tag.canonical_name,
                "facet": tag.facet,
                "value": round(f.feature_value, 4),
                "confidence": round(f.calibrated_confidence, 3),
                "raw_confidence": round(f.raw_confidence, 3),
                "weight": round(f.information_weight, 3),
                "rationale": j.rationale if j else None,
                "evidence": [
                    {"doc": e.document_id, "quote": e.quote} for e in (j.evidence if j else [])
                ],
                "shortlist_reason": j.shortlist_reason if j else None,
            }
        )

    # ``uncertain`` judgments are shown too: hiding them would misrepresent how
    # confident the dataset really is.
    uncertain = [
        {"tag_id": j.tag_id, "name": tags_by_id[j.tag_id].canonical_name, "notes": j.notes}
        for j in judgments
        if j.decision == "uncertain" and j.tag_id in tags_by_id
    ][:12]

    # "Why similar" is grounded in facts we can actually point at: shared tags
    # and shared normalised metadata fields. We never narrate a cause from
    # vector proximity alone.
    nb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    my_tags = {f.tag_id for f in features}
    my_meta = neighbor_facts.get(c.company_id, set())
    for n in sorted(neighbors, key=lambda n: (n.space, n.rank)):
        # A neighbour that did not make it into the published set would be a
        # dangling reference in the browser.
        if n.neighbor_company_id not in order:
            continue
        if len(nb[n.space]) >= PUBLISHED_NEIGHBORS_PER_SPACE:
            continue
        # Only the id, the score and the shared facts. Name, one-liner and
        # batch already live in companies.json, which the client has loaded
        # before it can open a company at all; repeating them here duplicated
        # the whole index roughly sixty times over across the detail shards.
        entry: dict[str, Any] = {
            "id": n.neighbor_company_id,
            "score": round(n.similarity, 4),
        }
        # Explanations are scoped to the space that produced the match. Shared
        # metadata explains a *metadata* neighbour; attaching it to a
        # description match would suggest the shared batch caused the
        # similarity, which is exactly the false causal story to avoid -- and
        # across five spaces it also dominated the payload.
        if n.space in ("combined", "tags", "sparse_tags"):
            shared_tags = sorted(my_tags & neighbor_tags.get(n.neighbor_company_id, set()))
            if shared_tags:
                entry["shared_tags"] = [
                    tags_by_id[t].canonical_name for t in shared_tags[:6] if t in tags_by_id
                ]
        if n.space in ("combined", "metadata"):
            shared_meta = sorted(my_meta & neighbor_facts.get(n.neighbor_company_id, set()))
            if shared_meta:
                entry["shared_metadata"] = shared_meta[:5]
        nb[n.space].append(entry)

    point = points_by_company.get(c.company_id)
    return {
        "company_id": c.company_id,
        "name": c.name,
        "one_liner": c.one_liner,
        "long_description": c.long_description,
        "website": c.website,
        "yc_url": c.yc_url,
        "logo_url": c.logo_url,
        "batch": c.batch,
        "batch_year": c.batch_year,
        "status": c.status,
        "stage": c.stage,
        "team_size": c.team_size,
        "industry": c.industry,
        "subindustry": c.subindustry,
        "industries": c.industries,
        "source_tags": c.source_tags,
        "regions": c.regions,
        "all_locations": c.all_locations,
        "former_names": c.former_names,
        "top_company": c.top_company,
        "is_hiring": c.is_hiring,
        "nonprofit": c.nonprofit,
        "metadata_document": c.metadata_document,
        "source_taxonomy_term_ids": c.source_taxonomy_term_ids,
        "tags": tag_rows,
        "uncertain_tags": uncertain,
        "neighbors": dict(nb),
        "coordinates": {"x": point.x, "y": point.y, "cluster": point.cluster_id} if point else None,
    }


def _metadata_facts(c: CompanyNormalized) -> set[str]:
    """Normalised metadata fields used to explain a metadata-similarity match."""
    facts: set[str] = set()
    if c.batch:
        facts.add(f"batch: {c.batch}")
    if c.industry:
        facts.add(f"industry: {c.industry}")
    if c.subindustry:
        facts.add(f"subindustry: {c.subindustry}")
    if c.status:
        facts.add(f"status: {c.status}")
    if c.stage:
        facts.add(f"stage: {c.stage}")
    for r in c.regions:
        facts.add(f"region: {r}")
    for t in c.source_tags:
        facts.add(f"yc tag: {t}")
    return facts
