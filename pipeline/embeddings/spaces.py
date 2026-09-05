"""The four dense representations, and how they are combined.

Every representation is produced by the *same* embedding model and version, so
they live in one comparable space. That is what makes the combined vector
legitimate: we take a weighted sum of L2-normalised component vectors and
renormalise, rather than concatenating vectors from incompatible spaces.

The alternative -- embedding one concatenated document -- was rejected because
it destroys the ability to serve description-only or metadata-only similarity
without a second pass over the model. Component vectors are all preserved.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import EmbeddingConfig
from pipeline.models import CompanyTagFeature, Tag
from pipeline.util import stable_hash
from pipeline.versions import COMBINED_TEMPLATE_VERSION


def embedding_space_version(model: str, digest: str | None, config: EmbeddingConfig) -> str:
    """Identity of the vector space.

    Changing the model, its weights, or the combination weights produces a new
    space version; neighbours from different space versions are never mixed.
    """
    return (
        "emb-"
        + stable_hash(
            {
                "model": model,
                "digest": digest,
                "template": COMBINED_TEMPLATE_VERSION,
                "w_description": config.weight_description,
                "w_metadata": config.weight_metadata,
                "w_tags": config.weight_tags,
            }
        )[:12]
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / n if n > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def tag_document_for_company(
    features: list[CompanyTagFeature], tags_by_id: dict[str, Tag], *, limit: int = 24
) -> str:
    """Canonical text for ``tag_embedding``.

    Tags are ordered by feature value and repeated proportionally to confidence
    only in the sense that the strongest ones come first and weak ones are
    truncated; the definition text carries the semantics.
    """
    ranked = sorted(features, key=lambda f: -f.feature_value)[:limit]
    parts = []
    for f in ranked:
        tag = tags_by_id.get(f.tag_id)
        if tag is None:
            continue
        parts.append(f"{tag.canonical_name}: {tag.definition}")
    return " ".join(parts)


def combine_vectors(
    description: np.ndarray | None,
    metadata: np.ndarray | None,
    tags: np.ndarray | None,
    config: EmbeddingConfig,
) -> np.ndarray:
    """Weighted sum of normalised components, renormalised.

    Missing components (a company with no description, or no positive tags) are
    dropped and the remaining weights are renormalised, so a sparse company is
    not pushed toward the origin and then amplified by normalisation noise.
    """
    parts: list[tuple[float, np.ndarray]] = []
    for weight, vec in (
        (config.weight_description, description),
        (config.weight_metadata, metadata),
        (config.weight_tags, tags),
    ):
        if vec is not None and weight > 0 and np.any(vec):
            parts.append((weight, l2_normalize(np.asarray(vec, dtype=np.float32))))
    if not parts:
        raise ValueError("cannot combine: no component vectors present")
    total = sum(w for w, _ in parts)
    stacked = np.zeros_like(parts[0][1])
    for weight, vec in parts:
        stacked += (weight / total) * vec
    return l2_normalize(stacked)


def quantize_int8(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantise unit vectors to int8 for the browser payload.

    Components of an L2-normalised high-dimensional vector are small, so a
    single global scale keeps the error far below the differences that matter
    for ranking. Full precision stays in Parquet.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    scale = float(np.max(np.abs(arr))) or 1.0
    return np.clip(np.round(arr / scale * 127.0), -127, 127).astype(np.int8), scale
