"""2D UMAP projection and algorithmic cluster labels.

Two honesty constraints shape this module:

* UMAP coordinates are **for visualisation only**. Nearest neighbours are always
  read from the precomputed high-dimensional lists, never from 2D distance.
  Nothing here writes neighbours.
* Coordinates move when the corpus changes. We seed the run, record every
  parameter in ``projection_version``, and optionally align a new projection to
  the previous release with a similarity transform fitted on shared companies,
  so a batch-over-batch comparison is not dominated by an arbitrary rotation.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import ProjectionConfig
from pipeline.models import Cluster, CompanyTagFeature, Tag, UmapPoint
from pipeline.util import log, stable_hash

LOG = log(__name__)


def projection_version(config: ProjectionConfig, embedding_space_version: str, n: int) -> str:
    return (
        "umap-"
        + stable_hash(
            {
                "n_neighbors": config.n_neighbors,
                "min_dist": config.min_dist,
                "metric": config.metric,
                "seed": config.seed,
                "space": embedding_space_version,
                "n": n,
            }
        )[:12]
    )


def project_umap(
    company_ids: list[str],
    matrix: np.ndarray,
    config: ProjectionConfig,
    *,
    embedding_space_version: str,
    previous: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[UmapPoint], np.ndarray]:
    """Fit UMAP and KMeans. Returns points plus the raw 2D coordinates."""
    import umap
    from sklearn.cluster import KMeans

    n = len(company_ids)
    if n == 0:
        return [], np.zeros((0, 2), dtype=np.float32)
    version = projection_version(config, embedding_space_version, n)

    if n < 5:
        # UMAP is undefined for a handful of points; a deterministic fallback
        # keeps the fixture profile working without special-casing the UI.
        coords = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            angle = 2 * np.pi * i / max(1, n)
            coords[i] = (np.cos(angle), np.sin(angle))
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(config.n_neighbors, n - 1),
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.seed,
            # A fixed seed forces single-threaded layout, which is the price of
            # reproducibility; documented in the methodology page.
            n_jobs=1,
            verbose=False,
        )
        coords = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)

    if previous:
        coords = align_to_previous(company_ids, coords, previous)
    coords = _rescale(coords)

    n_clusters = max(1, min(config.n_clusters, n))
    if n_clusters > 1:
        km = KMeans(n_clusters=n_clusters, random_state=config.seed, n_init=10)
        labels = km.fit_predict(coords)
    else:
        labels = np.zeros(n, dtype=int)

    points = [
        UmapPoint(
            company_id=cid,
            x=round(float(coords[i, 0]), 4),
            y=round(float(coords[i, 1]), 4),
            cluster_id=int(labels[i]),
            projection_version=version,
            embedding_space_version=embedding_space_version,
        )
        for i, cid in enumerate(company_ids)
    ]
    return points, coords


def _rescale(coords: np.ndarray) -> np.ndarray:
    """Centre and scale to roughly [-1, 1] so the frontend viewport is stable."""
    if coords.size == 0:
        return coords
    centred = coords - coords.mean(axis=0)
    scale = float(np.percentile(np.abs(centred), 99)) or 1.0
    return (centred / scale).astype(np.float32)


def align_to_previous(
    company_ids: list[str],
    coords: np.ndarray,
    previous: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Fit a similarity transform (rotation + scale + translation) on shared companies.

    UMAP's output is only defined up to rotation and reflection, so a new
    release can look completely rearranged even when nothing moved
    semantically. Orthogonal Procrustes on the companies present in both
    releases removes that artefact without distorting relative positions.
    """
    shared = [(i, previous[cid]) for i, cid in enumerate(company_ids) if cid in previous]
    if len(shared) < 8:
        LOG.info("projection alignment skipped: only %d shared companies", len(shared))
        return coords
    idx = np.array([i for i, _ in shared])
    src = coords[idx]
    dst = np.array([p for _, p in shared], dtype=np.float32)

    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    u, s, vt = np.linalg.svd(src_c.T @ dst_c)
    rotation = u @ vt
    var = float((src_c**2).sum())
    scale = float(s.sum() / var) if var > 1e-12 else 1.0
    aligned = (coords - src.mean(axis=0)) @ rotation * scale + dst.mean(axis=0)
    LOG.info("projection aligned to previous release on %d shared companies", len(shared))
    return aligned.astype(np.float32)


def label_clusters(
    points: list[UmapPoint],
    features: list[CompanyTagFeature],
    tags_by_id: dict[str, Tag],
    company_names: dict[str, str],
    *,
    top_n: int = 5,
) -> list[Cluster]:
    """Name clusters from over-represented tags, using lift rather than raw count.

    Raw frequency would label every cluster with the most common tag in the
    corpus. Lift (in-cluster rate / global rate) surfaces what actually
    distinguishes the cluster.
    """
    by_cluster: dict[int, list[str]] = {}
    for p in points:
        by_cluster.setdefault(p.cluster_id, []).append(p.company_id)

    global_counts: dict[str, int] = {}
    per_company: dict[str, set[str]] = {}
    for f in features:
        global_counts[f.tag_id] = global_counts.get(f.tag_id, 0) + 1
        per_company.setdefault(f.company_id, set()).add(f.tag_id)
    total = max(1, len({p.company_id for p in points}))

    clusters: list[Cluster] = []
    for cluster_id, members in sorted(by_cluster.items()):
        local: dict[str, int] = {}
        for cid in members:
            for tag_id in per_company.get(cid, ()):
                local[tag_id] = local.get(tag_id, 0) + 1
        scored: list[tuple[float, str]] = []
        for tag_id, count in local.items():
            if count < max(2, len(members) * 0.12):
                continue
            lift = (count / len(members)) / max(1e-9, global_counts.get(tag_id, 1) / total)
            scored.append((lift * count, tag_id))
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = [tag_id for _, tag_id in scored[:top_n]]
        names = [tags_by_id[t].canonical_name for t in top if t in tags_by_id]
        xs = [p.x for p in points if p.cluster_id == cluster_id]
        ys = [p.y for p in points if p.cluster_id == cluster_id]
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                label=" · ".join(names[:3]) if names else f"Cluster {cluster_id}",
                size=len(members),
                top_tag_ids=top,
                centroid_x=round(sum(xs) / max(1, len(xs)), 4),
                centroid_y=round(sum(ys) / max(1, len(ys)), 4),
                projection_version=points[0].projection_version if points else "",
            )
        )
    _ = company_names  # reserved for LLM cluster labelling; see prompts.cluster_prompt
    return clusters
