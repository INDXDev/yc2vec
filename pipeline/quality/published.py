"""Gates that run against the published artifacts alone.

The deploy step has only what is committed: `data/public/`. It does not have
the intermediate tables, so it cannot check referential integrity between, say,
features and judgments. What it *can* do -- and what actually matters before
shipping -- is verify that the bundle a browser will download is internally
consistent and matches its own manifest.

These checks are deliberately the ones a consumer of the artifacts could run
themselves.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import orjson

from pipeline.quality.gates import GateResult
from pipeline.util import log, sha256_file
from pipeline.versions import PUBLIC_ARTIFACT_VERSION

LOG = log(__name__)


def _load(root: Path, name: str) -> Any:
    return orjson.loads((root / name).read_bytes())


def _guard(name: str):
    """Turn an exception inside a check into a failing gate.

    A gate that raises on a malformed artifact is worse than no gate: the
    deploy job reports a crash instead of the actual problem, and a reader
    cannot tell a broken release from a broken checker. Every check therefore
    reports its own failure.
    """

    def decorator(fn):
        def wrapper(*args: Any, **kwargs: Any) -> GateResult:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - any failure is a gate failure
                return GateResult(name, False, f"{type(exc).__name__}: {exc}")

        return wrapper

    return decorator


def run_published_gates(public_dir: Path) -> list[GateResult]:
    root = public_dir / f"v{PUBLIC_ARTIFACT_VERSION}"
    results: list[GateResult] = []

    if not (root / "manifest.json").exists():
        return [GateResult("published release exists", False, f"no manifest under {root}")]

    try:
        manifest = _load(root, "manifest.json")
    except Exception as exc:  # noqa: BLE001
        return [GateResult("the manifest parses", False, f"{type(exc).__name__}: {exc}")]

    # 1. Every file the manifest claims is present and matches its checksum.
    bad: list[str] = []
    for rel, digest in (manifest.get("checksums") or {}).items():
        path = root / rel
        if not path.exists():
            bad.append(f"{rel} missing")
        elif sha256_file(path)[:16] != digest:
            bad.append(f"{rel} checksum mismatch")
    results.append(
        GateResult(
            "every published file matches its manifest checksum",
            not bad,
            f"{len(manifest.get('checksums') or {})} files verified" if not bad else "",
            bad[:5],
        )
    )

    @_guard("the published artifacts parse")
    def _read() -> GateResult:
        return GateResult("the published artifacts parse", True, "")

    try:
        points = _load(root, "points.json")
        companies = _load(root, "companies.json")
        tags = _load(root, "tags.json")
        company_ids = {row["i"] for row in companies["rows"]}
        tag_ids = {row["tag_id"] for row in tags["rows"]}
    except Exception as exc:  # noqa: BLE001
        results.append(
            GateResult("the published artifacts parse", False, f"{type(exc).__name__}: {exc}")
        )
        for r in results:
            LOG.info("published gate %-56s %s %s", r.name, "PASS" if r.passed else "FAIL", r.detail)
        return results
    _ = _read

    # 2. The map and the index describe exactly the same companies. A mismatch
    #    means a point with no detail, or a row that can never be plotted.
    point_ids = set(points["ids"])
    results.append(
        GateResult(
            "the map and the company index agree on who exists",
            point_ids == company_ids,
            f"{len(point_ids)} points, {len(company_ids)} companies",
            sorted(point_ids ^ company_ids)[:5],
        )
    )

    # 3. Coordinate arrays are parallel and finite.
    @_guard("coordinate arrays are parallel and finite")
    def _coords() -> GateResult:
        n = points["count"]
        lengths = {k: len(points[k]) for k in ("ids", "x", "y", "cluster", "year")}
        # JSON has no NaN or Infinity, and orjson writes them as null rather
        # than refusing -- so a non-finite coordinate reaches the browser as a
        # null that breaks the plot silently. Check for both.
        coords = points["x"] + points["y"]
        finite = all(isinstance(v, int | float) and math.isfinite(v) for v in coords)
        return GateResult(
            "coordinate arrays are parallel and finite",
            set(lengths.values()) == {n} and finite,
            f"count={n} lengths={lengths} finite={finite}",
        )

    results.append(_coords())

    # 4. Every tag a company references is published, or the UI renders a blank.
    dangling_tags = sorted(
        {t for row in companies["rows"] for t in row.get("T", []) if t not in tag_ids}
    )
    results.append(
        GateResult("every referenced tag is published", not dangling_tags, "", dangling_tags[:5])
    )

    # 5. Tag scores line up with tag ids.
    misaligned = [
        row["i"] for row in companies["rows"] if len(row.get("T", [])) != len(row.get("S", []))
    ]
    results.append(
        GateResult("tag ids and scores are the same length", not misaligned, "", misaligned[:5])
    )

    # 6. Detail shards cover every company, land in the right shard, and their
    #    neighbours point at companies that exist.
    from pipeline.publish.browser import shard_for

    covered: set[str] = set()
    misplaced: list[str] = []
    dangling_neighbors: list[str] = []
    shard_error: str | None = None
    try:
        for shard_path in sorted((root / "detail").glob("*.json")):
            shard_id = int(shard_path.stem)
            for cid, record in _load(root, f"detail/{shard_path.name}").items():
                covered.add(cid)
                if shard_for(cid) != shard_id:
                    misplaced.append(cid)
                for entries in (record.get("neighbors") or {}).values():
                    for entry in entries:
                        if entry.get("id") not in company_ids:
                            dangling_neighbors.append(f"{cid}->{entry.get('id')}")
    except Exception as exc:  # noqa: BLE001
        shard_error = f"{type(exc).__name__}: {exc}"

    results.append(
        GateResult(
            "every company has a detail record in its own shard",
            shard_error is None and covered == company_ids and not misplaced,
            shard_error or f"{len(covered)} records",
            (sorted(company_ids - covered)[:3] + misplaced[:2]),
        )
    )
    results.append(
        GateResult(
            "every published neighbour refers to a published company",
            shard_error is None and not dangling_neighbors,
            shard_error or "",
            dangling_neighbors[:5],
        )
    )

    # 7. The search index covers the corpus, or search silently misses companies.
    search_ids = {row["i"] for row in _load(root, "search/docs.json")["rows"]}
    results.append(
        GateResult(
            "the search index covers every published company",
            search_ids == company_ids,
            f"{len(search_ids)} documents",
            sorted(company_ids ^ search_ids)[:5],
        )
    )

    # 8. The manifest carries the provenance a consumer needs.
    required = ("dataset_version", "schema_version", "generated_at", "source_url", "models")
    missing = [f for f in required if not manifest.get(f)]
    results.append(
        GateResult(
            "the manifest records versions, source and models",
            not missing and bool(manifest.get("limitations")),
            "",
            missing,
        )
    )

    for r in results:
        LOG.info("published gate %-56s %s %s", r.name, "PASS" if r.passed else "FAIL", r.detail)
    return results
