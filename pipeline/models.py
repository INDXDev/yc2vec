"""Typed records for every table the pipeline reads or writes.

These Pydantic models are the single source of truth for the data contracts
listed in the concept document. JSON Schema is generated from them
(``yc2vec schemas``) and shared with the frontend's TypeScript types, so a
schema drift shows up as a type error rather than a silent runtime surprise.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["yes", "no", "uncertain"]
TagState = Literal["candidate", "active", "merged", "deprecated"]
EmbeddingKind = Literal["description", "metadata", "tags", "combined"]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------
# Source layer
# --------------------------------------------------------------------------


class CompanyRaw(Base):
    """A source record, preserved verbatim alongside its retrieval provenance."""

    company_id: str = Field(description="Stable YC2Vec id, e.g. 'ycoss:5'.")
    source: str = "yc_oss_api"
    source_url: str
    retrieved_at: datetime
    source_last_updated: str | None = None
    payload: dict[str, Any]
    content_hash: str


class SourceTaxonomyTerm(Base):
    """An exact YC/yc-oss classification, never renamed or merged by us."""

    term_id: str = Field(description="'{kind}:{slug}', e.g. 'industry:analytics'.")
    kind: Literal["industry", "subindustry", "tag", "batch", "region", "stage", "status"]
    slug: str
    name: str
    parent_term_id: str | None = None
    company_count: int = 0
    source_path: str | None = None
    retrieved_at: datetime


class WebSource(Base):
    """One fetched page. Bodies stay in the local cache, never in git."""

    company_id: str
    url: str
    final_url: str
    fetched_at: datetime
    status: int
    content_type: str | None = None
    content_hash: str | None = None
    byte_size: int = 0
    extraction_version: str
    ok: bool = False
    error: str | None = None
    robots_allowed: bool = True


class SourceDocument(Base):
    """Extracted, sanitised text that downstream stages may quote as evidence."""

    document_id: str
    company_id: str
    kind: Literal["yc_one_liner", "yc_long_description", "website_main_text", "metadata_document"]
    text: str
    char_count: int
    source_url: str | None = None
    content_hash: str
    extraction_version: str
    created_at: datetime


class CompanyNormalized(Base):
    """Cleaned, typed metadata used for exact filtering and for embedding."""

    company_id: str
    yc_id: int
    slug: str
    name: str
    former_names: list[str] = Field(default_factory=list)
    one_liner: str | None = None
    long_description: str | None = None
    website: str | None = None
    yc_url: str
    logo_url: str | None = None
    batch: str | None = None
    batch_slug: str | None = None
    batch_season: Literal["Winter", "Summer", "Spring", "Fall", "Unspecified"] = "Unspecified"
    batch_year: int | None = None
    status: str | None = None
    stage: str | None = None
    team_size: int | None = None
    is_hiring: bool = False
    nonprofit: bool = False
    top_company: bool = False
    industry: str | None = None
    subindustry: str | None = None
    industries: list[str] = Field(default_factory=list)
    source_tags: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    all_locations: str | None = None
    launched_at: datetime | None = None
    source_taxonomy_term_ids: list[str] = Field(default_factory=list)
    metadata_document: str = Field(
        description="Deterministic NL serialisation of the fields above."
    )
    metadata_template_version: str
    normalize_version: str
    content_hash: str
    updated_at: datetime


# --------------------------------------------------------------------------
# Ontology
# --------------------------------------------------------------------------


class TagCandidate(Base):
    candidate_id: str
    proposed_name: str
    normalized_name: str
    facet: str
    definition: str
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    support_company_ids: list[str] = Field(default_factory=list)
    discovery_run_id: str
    model: str
    prompt_version: str
    created_at: datetime
    resolution: Literal["pending", "activated", "merged", "rejected"] = "pending"
    resolved_tag_id: str | None = None


class Tag(Base):
    tag_id: str = Field(description="Stable slug, independent of display name.")
    canonical_name: str
    definition: str
    facet: str
    aliases: list[str] = Field(default_factory=list)
    normalized_aliases: list[str] = Field(default_factory=list)
    parent_tag_ids: list[str] = Field(default_factory=list)
    related_tag_ids: list[str] = Field(default_factory=list)
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    state: TagState = "candidate"
    merged_into: str | None = None
    deprecation_reason: str | None = None
    support_count: int = 0
    proposer: str = "llm"
    source_company_ids: list[str] = Field(default_factory=list)
    discovery_run_id: str | None = None
    prompt_version: str | None = None
    model: str | None = None
    ontology_version: str
    created_at: datetime
    updated_at: datetime


class TagAlias(Base):
    alias_id: str
    tag_id: str
    alias: str
    normalized_alias: str
    origin: Literal["llm", "merge", "manual"]
    created_at: datetime


class MergeProposal(Base):
    """An ambiguous merge waiting for review. Merges never happen silently."""

    proposal_id: str
    source_tag_id: str
    target_tag_id: str
    similarity: float
    verdict: Literal["auto_merge", "review", "distinct"]
    adjudication: Literal["merge", "distinct", "unclear"] | None = None
    rationale: str | None = None
    model: str | None = None
    created_at: datetime
    applied: bool = False


class SourceTaxonomyTagMapping(Base):
    """How a YC/yc-oss term relates to a YC2Vec tag. Reviewed, never implicit."""

    mapping_id: str
    term_id: str
    tag_id: str
    relation: Literal["equivalent", "broader", "narrower", "overlaps", "related"]
    similarity: float
    method: Literal["embedding", "llm", "manual"]
    reviewed: bool = False
    created_at: datetime


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


class EvidenceSpan(Base):
    document_id: str
    quote: str


class CompanyTagJudgment(Base):
    """One independent company/tag decision, with everything needed to audit it."""

    judgment_id: str
    company_id: str
    tag_id: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    notes: str | None = Field(default=None, description="Contradictions or missing information.")
    shortlist_reason: Literal[
        "retrieval", "facet_prior", "alias", "metadata_rule", "parent", "hard_negative"
    ]
    retrieval_score: float | None = None
    model: str
    model_digest: str | None = None
    prompt_version: str
    ontology_version: str
    pipeline_version: str
    run_id: str
    created_at: datetime


class CompanyTagFeature(Base):
    """A published positive assignment: the sparse company x tag vector, long form."""

    company_id: str
    tag_id: str
    present: int = Field(description="Binary presence, unweighted.")
    raw_confidence: float
    calibrated_confidence: float
    information_weight: float
    feature_value: float = Field(description="calibrated_confidence * information_weight")
    judgment_id: str
    ontology_version: str
    run_id: str


# --------------------------------------------------------------------------
# Vectors and projection
# --------------------------------------------------------------------------


class CompanyEmbedding(Base):
    company_id: str
    kind: EmbeddingKind
    dim: int
    embedding_space_version: str
    model: str
    model_digest: str | None = None
    template_version: str
    document_hash: str
    #: L2-normalised float32.
    vector: list[float]
    created_at: datetime


class Neighbor(Base):
    company_id: str
    neighbor_company_id: str
    space: Literal["combined", "description", "metadata", "tags", "sparse_tags"]
    rank: int
    similarity: float
    embedding_space_version: str


class UmapPoint(Base):
    company_id: str
    x: float
    y: float
    cluster_id: int
    projection_version: str
    embedding_space_version: str


class Cluster(Base):
    cluster_id: int
    label: str = Field(description="Algorithmic label from over-represented tags or source terms.")
    #: Which signal produced the label, so the UI can say what it is describing.
    label_source: Literal["semantic_tags", "source_taxonomy", "none"] = "none"
    size: int
    top_tag_ids: list[str] = Field(default_factory=list)
    top_source_terms: list[str] = Field(default_factory=list)
    centroid_x: float
    centroid_y: float
    projection_version: str


# --------------------------------------------------------------------------
# Runs and releases
# --------------------------------------------------------------------------


class PipelineRun(Base):
    run_id: str
    stage: str
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["running", "ok", "failed", "interrupted"] = "running"
    profile: str
    pipeline_version: str
    schema_version: str
    config_fingerprint: str
    git_commit: str | None = None
    models: dict[str, Any] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class ReleaseManifest(Base):
    dataset_version: str
    schema_version: str
    public_artifact_version: str
    pipeline_version: str
    ontology_version: str
    embedding_space_version: str
    projection_version: str
    generated_at: datetime
    git_commit: str | None = None
    source_retrieved_at: datetime | None = None
    source_last_updated: str | None = None
    source_url: str
    models: dict[str, Any] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    licenses: list[dict[str, str]] = Field(default_factory=list)
    attribution: str = ""


ALL_MODELS: dict[str, type[BaseModel]] = {
    "companies_raw": CompanyRaw,
    "companies_normalized": CompanyNormalized,
    "web_sources": WebSource,
    "source_documents": SourceDocument,
    "tags": Tag,
    "tag_aliases": TagAlias,
    "tag_candidates": TagCandidate,
    "merge_proposals": MergeProposal,
    "source_taxonomy_terms": SourceTaxonomyTerm,
    "source_taxonomy_tag_mappings": SourceTaxonomyTagMapping,
    "company_tag_judgments": CompanyTagJudgment,
    "company_tag_features": CompanyTagFeature,
    "company_embeddings": CompanyEmbedding,
    "company_neighbors": Neighbor,
    "umap_points": UmapPoint,
    "clusters": Cluster,
    "pipeline_runs": PipelineRun,
    "release_manifest": ReleaseManifest,
}
