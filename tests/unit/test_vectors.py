"""Vector semantics: dimensions, normalisation, missing values, neighbours."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import EmbeddingConfig, ProjectionConfig
from pipeline.embeddings.neighbors import sparse_neighbors, top_k_neighbors
from pipeline.embeddings.spaces import (
    combine_vectors,
    embedding_space_version,
    l2_normalize,
    quantize_int8,
)
from pipeline.projection.umap_project import align_to_previous, project_umap, projection_version

CFG = EmbeddingConfig()


def unit(*values: float) -> np.ndarray:
    return l2_normalize(np.asarray(values, dtype=np.float32))


def test_l2_normalize_produces_unit_vectors():
    m = l2_normalize(np.random.default_rng(0).normal(size=(20, 16)))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_l2_normalize_leaves_zero_vectors_alone():
    """A zero vector has no direction; scaling it would manufacture one."""
    z = l2_normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.all(z == 0)
    assert not np.any(np.isnan(z))


def test_combined_vector_is_unit_and_weighted():
    d, m, t = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)
    combined = combine_vectors(d, m, t, CFG)
    assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-5)
    # Ordering follows the configured weights: description > tags > metadata.
    assert combined[0] > combined[2] > combined[1]


def test_missing_components_renormalise_rather_than_shrink():
    d = unit(1, 0, 0)
    only_description = combine_vectors(d, None, None, CFG)
    assert np.allclose(only_description, d, atol=1e-6)
    assert np.isclose(np.linalg.norm(only_description), 1.0, atol=1e-5)

    # A company with no tags must not be pulled toward the origin.
    no_tags = combine_vectors(d, unit(0, 1, 0), None, CFG)
    assert np.isclose(np.linalg.norm(no_tags), 1.0, atol=1e-5)


def test_zero_tag_vector_is_treated_as_missing():
    d, m = unit(1, 0, 0), unit(0, 1, 0)
    zeros = np.zeros(3, dtype=np.float32)
    assert np.allclose(combine_vectors(d, m, zeros, CFG), combine_vectors(d, m, None, CFG))


def test_combining_nothing_is_an_error_not_a_zero_vector():
    with pytest.raises(ValueError):
        combine_vectors(None, None, None, CFG)


def test_embedding_space_version_tracks_model_and_weights():
    from dataclasses import replace

    base = embedding_space_version("m", "digest", CFG)
    assert base == embedding_space_version("m", "digest", CFG)
    assert base != embedding_space_version("m2", "digest", CFG)
    assert base != embedding_space_version("m", "other-digest", CFG)
    assert base != embedding_space_version("m", "digest", replace(CFG, weight_tags=0.9))


def test_top_k_neighbors_are_exact_and_exclude_self():
    rng = np.random.default_rng(3)
    ids = [f"c{i}" for i in range(40)]
    m = l2_normalize(rng.normal(size=(40, 12)))
    neighbors = top_k_neighbors(ids, m, space="combined", k=5, embedding_space_version="v", block=7)

    assert len(neighbors) == 40 * 5
    for n in neighbors:
        assert n.company_id != n.neighbor_company_id
        assert -1.0001 <= n.similarity <= 1.0001

    # Cross-check the first company against a brute-force ranking.
    sims = m @ m[0]
    sims[0] = -np.inf
    expected = [ids[i] for i in np.argsort(-sims)[:5]]
    got = [n.neighbor_company_id for n in neighbors if n.company_id == "c0"]
    assert got == expected


def test_neighbor_ranks_are_monotonic():
    rng = np.random.default_rng(11)
    ids = [f"c{i}" for i in range(12)]
    m = l2_normalize(rng.normal(size=(12, 8)))
    neighbors = [
        n
        for n in top_k_neighbors(ids, m, space="combined", k=4, embedding_space_version="v")
        if n.company_id == "c3"
    ]
    assert [n.rank for n in neighbors] == [0, 1, 2, 3]
    scores = [n.similarity for n in neighbors]
    assert scores == sorted(scores, reverse=True)


def test_sparse_neighbors_use_tag_overlap():
    rows = {
        "a": {"t1": 1.0, "t2": 1.0},
        "b": {"t1": 1.0, "t2": 1.0},  # identical profile
        "c": {"t1": 1.0},  # partial overlap
        "d": {"t9": 1.0},  # disjoint
    }
    result = sparse_neighbors(["a"], rows, k=3, embedding_space_version="v")
    ranked = [n.neighbor_company_id for n in result]
    assert ranked[0] == "b"
    assert "d" not in ranked  # no shared tag means no similarity at all
    assert result[0].similarity == pytest.approx(1.0)


def test_sparse_neighbors_skip_untagged_companies():
    assert sparse_neighbors(["x"], {"x": {}}, k=3, embedding_space_version="v") == []


def test_quantization_preserves_ranking():
    rng = np.random.default_rng(5)
    m = l2_normalize(rng.normal(size=(30, 64)))
    q, scale = quantize_int8(m)
    assert q.dtype == np.int8
    restored = q.astype(np.float32) * scale / 127.0
    exact = np.argsort(-(m @ m[0]))[:5]
    approx = np.argsort(-(restored @ restored[0]))[:5]
    assert set(exact) == set(approx)


def test_umap_projection_is_reproducible():
    rng = np.random.default_rng(1)
    ids = [f"c{i}" for i in range(60)]
    m = l2_normalize(rng.normal(size=(60, 10)))
    cfg = ProjectionConfig(n_neighbors=8, n_clusters=3)

    a, _ = project_umap(ids, m, cfg, embedding_space_version="v")
    b, _ = project_umap(ids, m, cfg, embedding_space_version="v")
    # Seeded UMAP must land in the same place within a documented tolerance.
    for pa, pb in zip(a, b, strict=True):
        assert abs(pa.x - pb.x) < 1e-3
        assert abs(pa.y - pb.y) < 1e-3
        assert pa.cluster_id == pb.cluster_id
    assert all(np.isfinite([p.x, p.y]).all() for p in a)


def test_projection_version_tracks_parameters():
    cfg = ProjectionConfig()
    from dataclasses import replace

    assert projection_version(cfg, "v", 10) == projection_version(cfg, "v", 10)
    assert projection_version(cfg, "v", 10) != projection_version(replace(cfg, seed=1), "v", 10)
    assert projection_version(cfg, "v", 10) != projection_version(cfg, "other", 10)


def test_small_corpora_do_not_crash_umap():
    ids = ["a", "b", "c"]
    m = l2_normalize(np.eye(3, dtype=np.float32))
    points, coords = project_umap(ids, m, ProjectionConfig(), embedding_space_version="v")
    assert len(points) == 3
    assert np.isfinite(coords).all()


def test_alignment_removes_rotation_between_releases():
    rng = np.random.default_rng(2)
    ids = [f"c{i}" for i in range(30)]
    previous = {
        cid: (float(x), float(y)) for cid, (x, y) in zip(ids, rng.normal(size=(30, 2)), strict=True)
    }

    # Simulate a rerun that produced the same structure, rotated 90 degrees.
    prev = np.array([previous[c] for c in ids], dtype=np.float32)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    rotated = prev @ rotation

    aligned = align_to_previous(ids, rotated, previous)
    assert np.allclose(aligned, prev, atol=1e-4)


def test_alignment_is_skipped_without_enough_shared_companies():
    coords = np.ones((4, 2), dtype=np.float32)
    out = align_to_previous(["a", "b", "c", "d"], coords, {"a": (0.0, 0.0)})
    assert np.array_equal(out, coords)
