"""Precomputed top-K nearest neighbours.

The site never computes similarity in the browser over full 4096-dimensional
vectors; it reads these precomputed lists. Cosine similarity on L2-normalised
vectors is a plain matrix product, computed in blocks so memory stays bounded
for the full corpus.
"""

from __future__ import annotations

import numpy as np

from pipeline.models import Neighbor


def top_k_neighbors(
    company_ids: list[str],
    matrix: np.ndarray,
    *,
    space: str,
    k: int,
    embedding_space_version: str,
    block: int = 512,
) -> list[Neighbor]:
    """Exact top-K by cosine similarity. ``matrix`` rows must be L2-normalised."""
    if len(company_ids) < 2:
        return []
    n = len(company_ids)
    k = min(k, n - 1)
    out: list[Neighbor] = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = matrix[start:stop] @ matrix.T
        # Exclude self-similarity before ranking.
        for row, i in enumerate(range(start, stop)):
            sims[row, i] = -np.inf
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for row, i in enumerate(range(start, stop)):
            order = idx[row][np.argsort(-sims[row, idx[row]])]
            for rank, j in enumerate(order):
                score = float(sims[row, j])
                if not np.isfinite(score):
                    continue
                out.append(
                    Neighbor(
                        company_id=company_ids[i],
                        neighbor_company_id=company_ids[int(j)],
                        space=space,  # type: ignore[arg-type]
                        rank=rank,
                        similarity=round(score, 6),
                        embedding_space_version=embedding_space_version,
                    )
                )
    return out


def sparse_neighbors(
    company_ids: list[str],
    rows: dict[str, dict[str, float]],
    *,
    k: int,
    embedding_space_version: str,
) -> list[Neighbor]:
    """Top-K in the sparse tag space, via weighted Jaccard over feature values.

    This deliberately uses a different metric from the dense spaces: it answers
    "which companies were given the same interpretable tags?", which is the
    question the tag-profile similarity mode exists to answer.
    """
    out: list[Neighbor] = []
    # Inverted index keeps this near-linear in the number of shared tags.
    postings: dict[str, list[str]] = {}
    for cid, tag_values in rows.items():
        for tag_id in tag_values:
            postings.setdefault(tag_id, []).append(cid)

    for cid in company_ids:
        feats: dict[str, float] | None = rows.get(cid)
        if not feats:
            continue
        candidates: set[str] = set()
        for tag_id in feats:
            candidates.update(postings.get(tag_id, ()))
        candidates.discard(cid)
        scored: list[tuple[float, str]] = []
        for other in candidates:
            other_feats = rows.get(other) or {}
            inter = 0.0
            union = 0.0
            for tag_id in feats.keys() | other_feats.keys():
                a = feats.get(tag_id, 0.0)
                b = other_feats.get(tag_id, 0.0)
                inter += min(a, b)
                union += max(a, b)
            if union > 0:
                scored.append((inter / union, other))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for rank, (score, other) in enumerate(scored[:k]):
            out.append(
                Neighbor(
                    company_id=cid,
                    neighbor_company_id=other,
                    space="sparse_tags",
                    rank=rank,
                    similarity=round(float(score), 6),
                    embedding_space_version=embedding_space_version,
                )
            )
    return out
