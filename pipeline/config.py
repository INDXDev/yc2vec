"""Runtime configuration.

Everything that changes pipeline behaviour lives here so that it can be
recorded in run manifests and used in cache keys. Values resolve in the order
CLI flag > environment variable > profile default, and the fully resolved
object is serialised into every ``pipeline_runs`` record.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

Profile = Literal["fixture", "balanced", "flagship"]

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Pinned upstream source. yc-oss/api republishes YC's public Algolia index.
YC_OSS_BASE_URL = "https://yc-oss.github.io/api"

USER_AGENT = (
    "YC2VecBot/0.1 (+https://github.com/INDXDev/yc2vec; "
    "unofficial open-data research project; contact via GitHub issues)"
)


@dataclass(frozen=True)
class ModelConfig:
    """Model selection. Never silently substituted -- ``doctor`` verifies it."""

    chat_model: str = "qwen3.8:27b"
    embedding_model: str = "qwen3-embedding:8b"
    #: Resolved at runtime by the Ollama client and recorded in manifests, so a
    #: release can be reproduced against the exact weights rather than a moving
    #: ``latest`` tag.
    chat_digest: str | None = None
    embedding_digest: str | None = None
    temperature: float = 0.0
    seed: int = 20240917
    num_ctx: int = 8192
    num_predict: int = 512
    #: Qwen3.8 is a hybrid-reasoning model; classification runs with thinking
    #: disabled for determinism and throughput.
    think: bool = False
    request_timeout_s: float = 300.0
    max_retries: int = 3


@dataclass(frozen=True)
class CrawlConfig:
    """Website enrichment. Opt-in, conservative and SSRF-hardened by default."""

    enabled: bool = False
    per_domain_concurrency: int = 1
    global_concurrency: int = 4
    request_delay_s: float = 1.0
    timeout_s: float = 20.0
    max_bytes: int = 1_500_000
    max_pages_per_company: int = 3
    max_redirects: int = 3
    respect_robots: bool = True
    #: Cached fetches younger than this are reused verbatim.
    cache_ttl_hours: int = 24 * 30
    allowed_content_types: tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
    )
    #: Same-origin paths worth a second look after the homepage.
    candidate_paths: tuple[str, ...] = ("/about", "/product", "/how-it-works", "/solutions")
    denylist_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class OntologyConfig:
    """Tag discovery, merge review and activation thresholds."""

    #: Soft target for the mature active ontology. Not a schema limit.
    target_active_tags: int = 1024
    discovery_batch_size: int = 12
    max_candidates_per_batch: int = 14
    #: Cosine similarity at or above which two tags merge without adjudication.
    auto_merge_threshold: float = 0.94
    #: Below this, tags are considered distinct without adjudication.
    review_threshold: float = 0.86
    #: Minimum number of proposing companies before a candidate can activate.
    min_support: int = 3
    facets: tuple[str, ...] = (
        "customer",
        "industry",
        "workflow",
        "business_model",
        "product_form",
        "technology",
        "data_modality",
        "buyer",
        "go_to_market",
        "deployment",
        "regulation",
        "geography",
        "company_stage",
        "problem_archetype",
    )


@dataclass(frozen=True)
class TaggingConfig:
    """Pair shortlisting and assignment."""

    #: Retrieval shortlist size per company, before hard negatives.
    shortlist_size: int = 24
    #: Calibrated hard negatives mixed into every company's candidate set so
    #: that precision is measurable rather than assumed.
    hard_negatives: int = 4
    #: Pairs judged per LLM call. 1 == strictly one call per pair. Grouping by
    #: facet keeps each judgement independent (own decision, confidence,
    #: rationale and evidence) while amortising the prompt across a facet.
    pairs_per_call: int = 8
    min_confidence: float = 0.55
    concurrency: int = 6
    #: Ubiquitous tags carry little information; the published feature value is
    #: ``calibrated_confidence * information_weight``.
    information_weight_floor: float = 0.15


@dataclass(frozen=True)
class EmbeddingConfig:
    batch_size: int = 16
    concurrency: int = 4
    #: Weights for the combined representation. Documented, versioned, tested.
    weight_description: float = 0.55
    weight_metadata: float = 0.20
    weight_tags: float = 0.25
    top_k_neighbors: int = 24
    #: Published dense vectors are quantised to int8 to keep the browser
    #: payload small; full-precision vectors stay in Parquet.
    publish_dense_vectors: bool = False


@dataclass(frozen=True)
class ProjectionConfig:
    n_neighbors: int = 25
    min_dist: float = 0.08
    metric: str = "cosine"
    seed: int = 20240917
    n_clusters: int = 24


@dataclass(frozen=True)
class Config:
    profile: Profile = "balanced"
    data_dir: Path = REPO_ROOT / "data"
    ollama_host: str = "http://localhost:11434"
    source_base_url: str = YC_OSS_BASE_URL
    user_agent: str = USER_AGENT
    models: ModelConfig = field(default_factory=ModelConfig)
    crawl: CrawlConfig = field(default_factory=CrawlConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    # -- derived layout -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def inferred_dir(self) -> Path:
        return self.data_dir / "inferred"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def public_dir(self) -> Path:
        return self.data_dir / "public"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "export"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def fingerprint(self) -> dict[str, Any]:
        """Configuration view used in cache keys and run manifests."""
        d = asdict(self)
        d.pop("data_dir", None)
        d["source_base_url"] = self.source_base_url
        return d


PROFILES: dict[str, dict[str, Any]] = {
    "fixture": {
        "models": ModelConfig(chat_model="fixture-chat", embedding_model="fixture-embed"),
        "ontology": OntologyConfig(discovery_batch_size=6, min_support=1),
        "tagging": TaggingConfig(shortlist_size=12, hard_negatives=2, concurrency=2),
        "embeddings": EmbeddingConfig(batch_size=8, concurrency=1, top_k_neighbors=8),
        "projection": ProjectionConfig(n_neighbors=5, n_clusters=4),
    },
    "balanced": {
        "models": ModelConfig(),
    },
    "flagship": {
        # An explicitly selected large local model. The doctor command refuses
        # to guess: the operator opts in and confirms the licence.
        "models": ModelConfig(chat_model="qwen3.8-flash-next:125b", num_ctx=16384),
        "tagging": TaggingConfig(shortlist_size=32, concurrency=4),
    },
}


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def load_config(
    profile: str | None = None,
    *,
    data_dir: Path | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    ollama_host: str | None = None,
    enable_crawl: bool | None = None,
) -> Config:
    """Resolve configuration from profile defaults, environment and CLI flags."""
    name = (profile or _env("YC2VEC_PROFILE", "balanced") or "balanced").lower()
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; expected one of {sorted(PROFILES)}")

    base = Config(profile=name, **PROFILES[name])  # type: ignore[arg-type]

    models = replace(
        base.models,
        chat_model=chat_model or _env("YC2VEC_CHAT_MODEL", base.models.chat_model) or "",
        embedding_model=(
            embedding_model or _env("YC2VEC_EMBEDDING_MODEL", base.models.embedding_model) or ""
        ),
        num_ctx=int(_env("YC2VEC_NUM_CTX", str(base.models.num_ctx)) or base.models.num_ctx),
        seed=int(_env("YC2VEC_SEED", str(base.models.seed)) or base.models.seed),
        think=_env_bool("YC2VEC_THINK", base.models.think),
    )
    crawl = replace(
        base.crawl,
        enabled=enable_crawl
        if enable_crawl is not None
        else _env_bool("YC2VEC_ENABLE_CRAWL", base.crawl.enabled),
        request_delay_s=float(
            _env("YC2VEC_CRAWL_DELAY", str(base.crawl.request_delay_s))
            or base.crawl.request_delay_s
        ),
    )
    return replace(
        base,
        data_dir=Path(
            data_dir or _env("YC2VEC_DATA_DIR", str(base.data_dir)) or base.data_dir
        ).resolve(),
        ollama_host=(
            ollama_host or _env("YC2VEC_OLLAMA_HOST", base.ollama_host) or base.ollama_host
        ).rstrip("/"),
        source_base_url=_env("YC2VEC_SOURCE_BASE_URL", base.source_base_url)
        or base.source_base_url,
        models=models,
        crawl=crawl,
    )
