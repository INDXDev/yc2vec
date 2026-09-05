"""CSV, Parquet and sparse-matrix exports for research use.

Parquet is the authoritative analytical store; CSV is a convenience export.
The company x tag relationship is exported *long form* (one row per
assignment) rather than as a 1,024-column matrix, because the wide form does
not scale with an open-ended ontology and hides the confidence and provenance
that make an assignment auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pipeline.models import (
    CompanyEmbedding,
    CompanyNormalized,
    CompanyTagFeature,
    CompanyTagJudgment,
    Neighbor,
    SourceTaxonomyTagMapping,
    SourceTaxonomyTerm,
    Tag,
    UmapPoint,
)
from pipeline.util import atomic_write, log

LOG = log(__name__)


def _frame(rows: Sequence[Any]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([r.model_dump(mode="json") for r in rows], strict=False)


def _write(df: pl.DataFrame, path: Path, *, csv: bool = False) -> int:
    if df.is_empty():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path.with_suffix(".parquet"), compression="zstd")
    if csv:
        # Nested columns have no faithful CSV representation; join them so the
        # export stays readable instead of emitting Python reprs.
        flat = df.with_columns(
            [
                pl.col(c).list.join("; ").alias(c)
                for c, dt in zip(df.columns, df.dtypes, strict=True)
                if dt == pl.List(pl.String)
            ]
        )
        drop = [
            c
            for c, dt in zip(flat.columns, flat.dtypes, strict=True)
            if isinstance(dt, pl.List | pl.Struct)
        ]
        flat.drop(drop).write_csv(path.with_suffix(".csv"))
    return df.height


def write_exports(
    export_dir: Path,
    *,
    companies: list[CompanyNormalized],
    tags: list[Tag],
    terms: list[SourceTaxonomyTerm],
    mappings: list[SourceTaxonomyTagMapping],
    features: list[CompanyTagFeature],
    judgments: list[CompanyTagJudgment],
    neighbors: list[Neighbor],
    points: list[UmapPoint],
    embeddings: list[CompanyEmbedding] | None = None,
) -> dict[str, int]:
    export_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    counts["companies"] = _write(_frame(companies), export_dir / "companies", csv=True)
    counts["tags"] = _write(_frame(tags), export_dir / "tags", csv=True)
    counts["source_taxonomy"] = _write(_frame(terms), export_dir / "source_taxonomy", csv=True)
    counts["source_taxonomy_tag_mappings"] = _write(
        _frame(mappings), export_dir / "source_taxonomy_tag_mappings", csv=True
    )
    counts["company_tags"] = _write(_frame(features), export_dir / "company_tags", csv=True)
    counts["company_tag_judgments"] = _write(
        _frame(judgments), export_dir / "company_tag_judgments", csv=False
    )
    counts["company_neighbors"] = _write(
        _frame(neighbors), export_dir / "company_neighbors", csv=False
    )
    counts["umap_points"] = _write(_frame(points), export_dir / "umap_points", csv=True)
    if embeddings:
        counts["company_embeddings"] = _write(
            _frame(embeddings), export_dir / "company_embeddings", csv=False
        )

    counts["matrix_nnz"] = write_sparse_matrix(
        export_dir / "company_tag_matrix.npz", companies, tags, features
    )
    LOG.info("exports written to %s (%s)", export_dir, counts)
    return counts


def write_sparse_matrix(
    path: Path,
    companies: list[CompanyNormalized],
    tags: list[Tag],
    features: list[CompanyTagFeature],
) -> int:
    """CSR company x tag matrix plus row/column maps, for research use."""
    from scipy import sparse

    company_ids = [c.company_id for c in companies]
    tag_ids = [t.tag_id for t in tags if t.state == "active"]
    row_index = {cid: i for i, cid in enumerate(company_ids)}
    col_index = {tid: i for i, tid in enumerate(tag_ids)}

    rows, cols, vals = [], [], []
    for f in features:
        r, c = row_index.get(f.company_id), col_index.get(f.tag_id)
        if r is None or c is None:
            continue
        rows.append(r)
        cols.append(c)
        vals.append(f.feature_value)
    if not vals:
        return 0

    matrix = sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (rows, cols)),
        shape=(len(company_ids), len(tag_ids)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path) as fh:
        np.savez_compressed(
            fh,
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=np.asarray(matrix.shape),
            company_ids=np.asarray(company_ids, dtype=object),
            tag_ids=np.asarray(tag_ids, dtype=object),
        )
    return int(matrix.nnz)
