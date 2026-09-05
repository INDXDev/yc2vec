"""Stage orchestration.

Each function here is one resumable stage. They share three properties:

* **Idempotent** -- rerunning with unchanged inputs writes identical output.
* **Checkpointed** -- long stages append to a partial file after every company,
  so an interrupt costs at most one unit of work.
* **Content-addressed** -- the store's manifest records a stage key derived from
  upstream hashes, config, prompts and model identity. A stage is skipped when
  its key is unchanged unless ``force`` is set.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import PIPELINE_VERSION
from pipeline.config import Config
from pipeline.embeddings.neighbors import sparse_neighbors, top_k_neighbors
from pipeline.embeddings.spaces import (
    combine_vectors,
    embedding_space_version,
    l2_normalize,
    tag_document_for_company,
)
from pipeline.fetch.stage import load_normalized, load_terms, load_website_texts
from pipeline.models import (
    Cluster,
    CompanyEmbedding,
    CompanyNormalized,
    CompanyTagFeature,
    CompanyTagJudgment,
    MergeProposal,
    Neighbor,
    PipelineRun,
    SourceTaxonomyTagMapping,
    Tag,
    UmapPoint,
)
from pipeline.normalize.companies import company_documents, description_document
from pipeline.ollama import OllamaClient
from pipeline.ontology.discovery import discover_candidates
from pipeline.ontology.mapping import map_source_taxonomy
from pipeline.ontology.merge import activate_candidates, apply_merge, embed_tags, propose_merges
from pipeline.ontology.registry import OntologyRegistry
from pipeline.projection.umap_project import label_clusters, project_umap
from pipeline.prompts import prompt_hashes
from pipeline.store import Store
from pipeline.tagging.assign import assign_many, build_features
from pipeline.tagging.shortlist import Shortlister
from pipeline.util import (
    append_jsonl,
    git_commit,
    log,
    now,
    read_jsonl,
    stable_hash,
    write_jsonl,
)
from pipeline.versions import METADATA_TEMPLATE_VERSION, ONTOLOGY_VERSION, SCHEMA_VERSION

LOG = log(__name__)


def stratified_sample(
    companies: list[CompanyNormalized], size: int, *, seed: int
) -> list[CompanyNormalized]:
    """Pick a representative subset, spread across industry and batch year.

    Assignment is the expensive stage, so a partial run is the normal case
    rather than a failure mode. Taking the first N companies would bias the
    result toward whatever the source happens to order first -- in practice the
    oldest batches. Sampling proportionally across (industry, batch year) keeps
    a partial dataset usable for comparison across the whole corpus, and the
    seed keeps the choice reproducible.
    """
    if size >= len(companies):
        return companies

    strata: dict[tuple[str, int], list[CompanyNormalized]] = {}
    for c in companies:
        strata.setdefault((c.industry or "Unlisted", c.batch_year or 0), []).append(c)

    rng = np.random.default_rng(seed)
    picked: list[CompanyNormalized] = []
    # Round-robin over strata so every stratum contributes before any
    # contributes twice; large strata still end up proportionally represented
    # because they simply have more rounds to give to.
    #
    # The *order* of the strata is shuffled as well as their contents. Many
    # strata hold a single company, and with those a fixed iteration order
    # would make the sample a deterministic prefix of the sorted keys -- which
    # in practice means one or two industries, exactly the bias this function
    # exists to avoid.
    pools = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda c: c.company_id)
        rng.shuffle(group)
        pools.append(group)
    rng.shuffle(pools)
    idx = 0
    while len(picked) < size and any(pools):
        pool = pools[idx % len(pools)]
        if pool:
            picked.append(pool.pop())
        idx += 1
        if idx % len(pools) == 0:
            pools = [p for p in pools if p]
            if not pools:
                break
            idx = 0
    return sorted(picked, key=lambda c: c.company_id)[:size]


def new_run(config: Config, stage: str, models: dict[str, Any] | None = None) -> PipelineRun:
    return PipelineRun(
        run_id=uuid.uuid4().hex,
        stage=stage,
        started_at=now(),
        profile=config.profile,
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        config_fingerprint=stable_hash(config.fingerprint()),
        git_commit=git_commit(),
        models=models or {},
        prompt_hashes=prompt_hashes(),
    )


# --------------------------------------------------------------------------
# Ontology
# --------------------------------------------------------------------------


async def discover_stage(
    config: Config,
    store: Store,
    client: OllamaClient,
    *,
    max_batches: int | None = None,
    limit: int | None = None,
    concurrency: int = 4,
) -> dict[str, int]:
    companies = load_normalized(store)
    if limit:
        companies = companies[:limit]
    registry = OntologyRegistry(store.path("inferred", "ontology"))
    run = new_run(config, "discover-tags", {"chat": config.models.chat_model})

    candidates = await discover_candidates(
        config=config,
        client=client,
        registry=registry,
        companies=companies,
        run_id=run.run_id,
        max_batches=max_batches,
        concurrency=concurrency,
    )

    # Fold candidates into tags, reusing an existing tag when the normalised
    # name or an alias already matches. Raw model strings never become ids.
    created = 0
    for candidate in candidates:
        if candidate.resolution != "pending":
            continue
        existing = registry.resolve(candidate.proposed_name)
        if existing is not None:
            existing.source_company_ids = sorted(
                set(existing.source_company_ids) | set(candidate.support_company_ids)
            )
            existing.support_count = len(existing.source_company_ids)
            existing.updated_at = now()
            candidate.resolution = "merged"
            candidate.resolved_tag_id = existing.tag_id
            continue
        tag = registry.create_tag(
            name=candidate.proposed_name,
            facet=candidate.facet,
            definition=candidate.definition,
            positive_examples=candidate.positive_examples,
            negative_examples=candidate.negative_examples,
            support_company_ids=candidate.support_company_ids,
            discovery_run_id=candidate.discovery_run_id,
            prompt_version=candidate.prompt_version,
            model=candidate.model,
        )
        candidate.resolution = "activated"
        candidate.resolved_tag_id = tag.tag_id
        created += 1

    registry.save()
    run.finished_at = now()
    run.status = "ok"
    run.counts = {"candidates": len(candidates), "new_tags": created, **registry.stats()}
    store.write_run(run)
    LOG.info("discover-tags: %s", run.counts)
    return run.counts


async def review_stage(
    config: Config,
    store: Store,
    client: OllamaClient,
    *,
    apply: bool = True,
    max_adjudications: int | None = 400,
    concurrency: int = 4,
) -> dict[str, int]:
    registry = OntologyRegistry(store.path("inferred", "ontology"))
    run = new_run(config, "review-tags", {"chat": config.models.chat_model})

    reviewable = [t for t in registry.tags.values() if t.state in ("candidate", "active")]
    proposals = await propose_merges(
        config=config,
        client=client,
        registry=registry,
        tags=reviewable,
        max_adjudications=max_adjudications,
        concurrency=concurrency,
    )

    applied = 0
    queued = 0
    for p in proposals:
        if apply and (p.verdict == "auto_merge" or p.adjudication == "merge"):
            applied += int(apply_merge(registry, p))
        elif p.verdict == "review" and p.adjudication not in ("merge", "distinct"):
            queued += 1

    activated = activate_candidates(registry, min_support=config.ontology.min_support)
    registry.save()
    write_jsonl(store.path("inferred", "ontology", "merge_proposals.jsonl"), proposals)

    run.finished_at = now()
    run.status = "ok"
    run.counts = {
        "proposals": len(proposals),
        "merges_applied": applied,
        "review_queue": queued,
        "activated": activated,
        **registry.stats(),
    }
    store.write_run(run)
    LOG.info("review-tags: %s", run.counts)
    return run.counts


async def map_taxonomy_stage(config: Config, store: Store, client: OllamaClient) -> dict[str, int]:
    registry = OntologyRegistry(store.path("inferred", "ontology"))
    terms = load_terms(store)
    active = registry.active()
    mappings = await map_source_taxonomy(
        client=client, terms=terms, tags=active, batch_size=config.embeddings.batch_size
    )
    write_jsonl(store.path("inferred", "source_taxonomy_tag_mappings.jsonl"), mappings)
    return {"mappings": len(mappings), "terms": len(terms), "tags": len(active)}


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


async def assign_stage(
    config: Config,
    store: Store,
    client: OllamaClient,
    *,
    limit: int | None = None,
    company_ids: list[str] | None = None,
    sample: int | None = None,
    resume: bool = True,
    finalize_only: bool = False,
) -> dict[str, int]:
    """Judge shortlisted pairs and build the sparse features.

    ``finalize_only`` consolidates whatever the checkpoint already holds into
    the final tables without judging anything new. Assignment is the expensive
    stage and is expected to be interrupted, so harvesting a partial run is a
    first-class operation rather than a recovery hack.
    """
    companies = load_normalized(store)
    if company_ids:
        wanted = set(company_ids)
        companies = [c for c in companies if c.company_id in wanted]
    if sample:
        companies = stratified_sample(companies, sample, seed=config.models.seed)
    if limit:
        companies = companies[:limit]

    registry = OntologyRegistry(store.path("inferred", "ontology"))
    tags = registry.active()

    if finalize_only:
        return _finalize_judgments(config, store, n_companies=len(companies))

    if not tags:
        LOG.warning(
            "assign-tags: ontology has no active tags; run discover-tags and review-tags first"
        )
        return {"judgments": 0, "features": 0}

    tags_by_id = {t.tag_id: t for t in tags}
    mappings = [
        SourceTaxonomyTagMapping(**r)
        for r in read_jsonl(store.path("inferred", "source_taxonomy_tag_mappings.jsonl"))
    ]
    run = new_run(config, "assign-tags", {"chat": config.models.chat_model})
    digest = await client.resolve(config.models.chat_model)

    LOG.info("assign-tags: embedding %d tag definitions", len(tags))
    tag_matrix = await embed_tags(client, tags, config.embeddings.batch_size)

    website_texts = load_website_texts(store)
    docs_by_company = {
        c.company_id: company_documents(c, website_texts.get(c.company_id)) for c in companies
    }
    LOG.info("assign-tags: embedding %d company descriptions", len(companies))
    desc_docs = [description_document(c, website_texts.get(c.company_id)) for c in companies]
    company_matrix = l2_normalize(
        np.asarray(await _embed_all(client, desc_docs, config), dtype=np.float32)
    )

    shortlister = Shortlister(
        tags,
        tag_matrix,
        mappings=mappings,
        shortlist_size=config.tagging.shortlist_size,
        hard_negatives=config.tagging.hard_negatives,
        seed=config.models.seed,
    )

    # Resume needs only the set of companies already judged; the judgments
    # themselves are re-read from the checkpoint at finalize time, so there is
    # no reason to hold a growing corpus of them in memory here.
    partial = store.path("inferred", "company_tag_judgments.partial.jsonl")
    done: set[str] = set()
    if resume and partial.exists():
        done = {row["company_id"] for row in read_jsonl(partial)}
        LOG.info("assign-tags: resuming, %d companies already judged", len(done))

    work = []
    for i, c in enumerate(companies):
        if c.company_id in done:
            continue
        shortlist = shortlister.shortlist(
            c, company_matrix[i], f"{c.name} {c.one_liner or ''} {c.long_description or ''}"
        )
        work.append((c, docs_by_company[c.company_id], shortlist))

    LOG.info(
        "assign-tags: %d companies to judge, ~%d pairs", len(work), sum(len(w[2]) for w in work)
    )

    def checkpoint(company: CompanyNormalized, judgments: list[CompanyTagJudgment]) -> None:
        # Append after every company: an interrupt costs one company, not the run.
        append_jsonl(partial, judgments)

    fresh = await assign_many(
        config=config,
        client=client,
        work=work,
        tags_by_id=tags_by_id,
        run_id=run.run_id,
        model_digest=digest,
        on_result=checkpoint,
    )

    counts = _finalize_judgments(config, store, n_companies=len(companies), run_id=run.run_id)
    counts["companies"] = len(companies)
    counts["fresh_judgments"] = len(fresh)

    run.finished_at = now()
    run.status = "ok"
    run.counts = counts
    store.write_run(run)
    LOG.info("assign-tags: %s", counts)
    return counts


def _finalize_judgments(
    config: Config, store: Store, *, n_companies: int, run_id: str = ""
) -> dict[str, int]:
    """Promote the append-only checkpoint into the durable judgment and feature tables.

    The checkpoint is append-only and can hold the same pair twice if a company
    was re-judged after a prompt or ontology change; the later record wins, so
    the tables always reflect the most recent judgment.
    """
    partial = store.path("inferred", "company_tag_judgments.partial.jsonl")
    by_pair: dict[tuple[str, str], CompanyTagJudgment] = {}
    for row in read_jsonl(partial):
        j = CompanyTagJudgment(**row)
        by_pair[(j.company_id, j.tag_id)] = j
    judgments = sorted(by_pair.values(), key=lambda j: (j.company_id, j.tag_id))

    judged_companies = {j.company_id for j in judgments}
    write_jsonl(store.path("inferred", "company_tag_judgments.jsonl"), judgments)

    # Tag prevalence -- and therefore the information weight -- is measured over
    # the companies actually judged, not over the whole corpus. Dividing by the
    # corpus size while only a subset has been judged would make every tag look
    # rare and inflate its weight.
    features = build_features(
        judgments,
        n_companies=max(1, len(judged_companies)),
        min_confidence=config.tagging.min_confidence,
        weight_floor=config.tagging.information_weight_floor,
        run_id=run_id,
    )
    write_jsonl(store.path("inferred", "company_tag_features.jsonl"), features)

    counts = {
        "judgments": len(judgments),
        "features": len(features),
        "judged_companies": len(judged_companies),
        "corpus_companies": n_companies,
    }
    LOG.info("finalize: %s", counts)
    return counts


async def _embed_all(client: OllamaClient, docs: list[str], config: Config) -> list[list[float]]:
    out: list[list[float]] = []
    size = config.embeddings.batch_size
    for i in range(0, len(docs), size):
        out.extend(await client.embed(docs[i : i + size]))
        if i and i % (size * 50) == 0:
            LOG.info("  embedded %d/%d", i, len(docs))
    return out


# --------------------------------------------------------------------------
# Embeddings, neighbours, projection
# --------------------------------------------------------------------------


async def embed_stage(
    config: Config, store: Store, client: OllamaClient, *, limit: int | None = None
) -> dict[str, int]:
    companies = load_normalized(store)
    if limit:
        companies = companies[:limit]
    registry = OntologyRegistry(store.path("inferred", "ontology"))
    tags_by_id = {t.tag_id: t for t in registry.tags.values()}
    features = [
        CompanyTagFeature(**r)
        for r in read_jsonl(store.path("inferred", "company_tag_features.jsonl"))
    ]
    by_company: dict[str, list[CompanyTagFeature]] = {}
    for f in features:
        by_company.setdefault(f.company_id, []).append(f)

    run = new_run(config, "embed", {"embedding": config.models.embedding_model})
    digest = await client.resolve(config.models.embedding_model)
    space = embedding_space_version(config.models.embedding_model, digest, config.embeddings)

    website_texts = load_website_texts(store)
    ids = [c.company_id for c in companies]

    LOG.info("embed: description documents (%d)", len(companies))
    desc = l2_normalize(
        np.asarray(
            await _embed_all(
                client,
                [description_document(c, website_texts.get(c.company_id)) for c in companies],
                config,
            ),
            dtype=np.float32,
        )
    )
    LOG.info("embed: metadata documents (%d)", len(companies))
    meta = l2_normalize(
        np.asarray(
            await _embed_all(client, [c.metadata_document for c in companies], config),
            dtype=np.float32,
        )
    )

    LOG.info("embed: tag-profile documents")
    tag_docs = [
        tag_document_for_company(by_company.get(c.company_id, []), tags_by_id) for c in companies
    ]
    # Companies with no positive tags get a zero vector rather than an
    # embedding of the empty string, so the combiner can drop the component.
    non_empty = [i for i, d in enumerate(tag_docs) if d.strip()]
    tag_matrix = np.zeros((len(companies), desc.shape[1]), dtype=np.float32)
    if non_empty:
        vectors = await _embed_all(client, [tag_docs[i] for i in non_empty], config)
        tag_matrix[non_empty] = l2_normalize(np.asarray(vectors, dtype=np.float32))

    combined = np.zeros_like(desc)
    for i in range(len(companies)):
        combined[i] = combine_vectors(
            desc[i],
            meta[i],
            tag_matrix[i] if i in set(non_empty) else None,
            config.embeddings,
        )

    records: list[CompanyEmbedding] = []
    matrices = {"description": desc, "metadata": meta, "tags": tag_matrix, "combined": combined}
    for kind, matrix in matrices.items():
        template = METADATA_TEMPLATE_VERSION if kind == "metadata" else ONTOLOGY_VERSION
        for i, c in enumerate(companies):
            if kind == "tags" and i not in set(non_empty):
                continue
            records.append(
                CompanyEmbedding(
                    company_id=c.company_id,
                    kind=kind,  # type: ignore[arg-type]
                    dim=int(matrix.shape[1]),
                    embedding_space_version=space,
                    model=config.models.embedding_model,
                    model_digest=digest,
                    template_version=template,
                    document_hash=stable_hash({"kind": kind, "id": c.company_id}),
                    vector=[round(float(v), 6) for v in matrix[i]],
                    created_at=now(),
                )
            )
    write_jsonl(store.path("inferred", "company_embeddings.jsonl"), records)
    arrays: dict[str, Any] = {
        "ids": np.asarray(ids, dtype=object),
        "space": np.asarray([space], dtype=object),
        **matrices,
    }
    with open(store.path("inferred", "embeddings.npz"), "wb") as fh:
        np.savez_compressed(fh, **arrays)

    LOG.info("embed: computing top-%d neighbours in 5 spaces", config.embeddings.top_k_neighbors)
    neighbors: list[Neighbor] = []
    for space_name, matrix in matrices.items():
        rows = ids
        mat = matrix
        if space_name == "tags":
            keep = [i for i in range(len(ids)) if np.any(matrix[i])]
            rows = [ids[i] for i in keep]
            mat = matrix[keep]
        neighbors.extend(
            top_k_neighbors(
                rows,
                mat,
                space=space_name,
                k=config.embeddings.top_k_neighbors,
                embedding_space_version=space,
            )
        )
    sparse_rows = {
        cid: {f.tag_id: f.feature_value for f in feats} for cid, feats in by_company.items()
    }
    neighbors.extend(
        sparse_neighbors(
            ids, sparse_rows, k=config.embeddings.top_k_neighbors, embedding_space_version=space
        )
    )
    write_jsonl(store.path("inferred", "company_neighbors.jsonl"), neighbors)

    run.finished_at = now()
    run.status = "ok"
    run.counts = {
        "companies": len(companies),
        "embeddings": len(records),
        "neighbors": len(neighbors),
        "dim": int(desc.shape[1]),
    }
    store.write_run(run)
    LOG.info("embed: %s (space=%s)", run.counts, space)
    return run.counts


def project_stage(config: Config, store: Store, *, align: bool = True) -> dict[str, int]:
    data = np.load(store.path("inferred", "embeddings.npz"), allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    matrix = np.asarray(data["combined"], dtype=np.float32)
    space = str(data["space"][0])

    previous: dict[str, tuple[float, float]] | None = None
    prev_path = store.path("inferred", "umap_points.jsonl")
    if align and prev_path.exists():
        previous = {r["company_id"]: (r["x"], r["y"]) for r in read_jsonl(prev_path)}

    run = new_run(config, "project")
    points, _ = project_umap(
        ids, matrix, config.projection, embedding_space_version=space, previous=previous
    )

    registry = OntologyRegistry(store.path("inferred", "ontology"))
    features = [
        CompanyTagFeature(**r)
        for r in read_jsonl(store.path("inferred", "company_tag_features.jsonl"))
    ]
    companies = {c.company_id: c.name for c in load_normalized(store)}
    clusters = label_clusters(
        points, features, {t.tag_id: t for t in registry.tags.values()}, companies
    )

    write_jsonl(prev_path, points)
    write_jsonl(store.path("inferred", "clusters.jsonl"), clusters)

    run.finished_at = now()
    run.status = "ok"
    run.counts = {"points": len(points), "clusters": len(clusters)}
    store.write_run(run)
    LOG.info("project: %s", run.counts)
    return run.counts


# --------------------------------------------------------------------------
# Loading helpers used by publish/validate
# --------------------------------------------------------------------------


def load_artifacts(store: Store) -> dict[str, Any]:
    registry = OntologyRegistry(store.path("inferred", "ontology"))
    return {
        "companies": load_normalized(store),
        "terms": load_terms(store),
        "registry": registry,
        "tags": sorted(registry.tags.values(), key=lambda t: t.tag_id),
        "mappings": [
            SourceTaxonomyTagMapping(**r)
            for r in read_jsonl(store.path("inferred", "source_taxonomy_tag_mappings.jsonl"))
        ],
        "features": [
            CompanyTagFeature(**r)
            for r in read_jsonl(store.path("inferred", "company_tag_features.jsonl"))
        ],
        "judgments": [
            CompanyTagJudgment(**r)
            for r in read_jsonl(store.path("inferred", "company_tag_judgments.jsonl"))
        ],
        "neighbors": [
            Neighbor(**r) for r in read_jsonl(store.path("inferred", "company_neighbors.jsonl"))
        ],
        "points": [UmapPoint(**r) for r in read_jsonl(store.path("inferred", "umap_points.jsonl"))],
        "clusters": [Cluster(**r) for r in read_jsonl(store.path("inferred", "clusters.jsonl"))],
        "merge_proposals": [
            MergeProposal(**r)
            for r in read_jsonl(store.path("inferred", "ontology", "merge_proposals.jsonl"))
        ],
    }


def load_embeddings(store: Store) -> list[CompanyEmbedding]:
    return [
        CompanyEmbedding(**r)
        for r in read_jsonl(store.path("inferred", "company_embeddings.jsonl"))
    ]


def _tags_for_publish(tags: list[Tag]) -> list[Tag]:
    return [t for t in tags if t.state == "active"]


def dataset_version(store: Store, artifacts: dict[str, Any]) -> str:
    return stable_hash(
        {
            "companies": len(artifacts["companies"]),
            "tags": len(_tags_for_publish(artifacts["tags"])),
            "features": len(artifacts["features"]),
            "points": len(artifacts["points"]),
            "manifest": store.stage_output_hashes("publish-data"),
        }
    )[:12]


def public_root(config: Config) -> Path:
    return config.public_dir
