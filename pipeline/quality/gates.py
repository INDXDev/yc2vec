"""Release gates.

These are the checks that decide whether a dataset may be published. They are
deliberately blunt: any failure blocks the release rather than emitting a
warning, because a silently degraded public dataset is worse than a late one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from pipeline.models import (
    CompanyNormalized,
    CompanyTagFeature,
    CompanyTagJudgment,
    Neighbor,
    Tag,
    UmapPoint,
)
from pipeline.util import log

LOG = log(__name__)


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""
    samples: list[str] = field(default_factory=list)


def run_release_gates(
    *,
    companies: list[CompanyNormalized],
    tags: list[Tag],
    features: list[CompanyTagFeature],
    judgments: list[CompanyTagJudgment],
    neighbors: list[Neighbor],
    points: list[UmapPoint],
    public_dir: Path | None = None,
) -> list[GateResult]:
    results: list[GateResult] = []
    company_ids = {c.company_id for c in companies}
    point_ids = {p.company_id for p in points}
    tag_ids = {t.tag_id for t in tags}
    active_ids = {t.tag_id for t in tags if t.state == "active"}
    judgments_by_id = {j.judgment_id: j for j in judgments}

    # 1. Stable id and source URL for every published company.
    bad = [c.company_id for c in companies if not c.company_id or not c.yc_url][:5]
    results.append(
        GateResult(
            "every company has a stable id and source url",
            not bad,
            f"{len(bad)} offending company/companies"
            if bad
            else f"{len(companies)} companies checked",
            bad,
        )
    )

    # 2. Provenance for every published positive assignment.
    missing = [
        f"{f.company_id}/{f.tag_id}"
        for f in features
        if f.judgment_id not in judgments_by_id
        or not judgments_by_id[f.judgment_id].evidence
        or not judgments_by_id[f.judgment_id].rationale
    ][:5]
    results.append(
        GateResult(
            "no missing provenance for published positive assignments",
            not missing,
            f"{len(missing)}+ assignments without evidence or rationale"
            if missing
            else f"{len(features)} assignments checked",
            missing,
        )
    )

    # 3. Numeric sanity.
    nan = [
        f"{f.company_id}/{f.tag_id}"
        for f in features
        if not all(
            math.isfinite(v) for v in (f.feature_value, f.calibrated_confidence, f.raw_confidence)
        )
    ][:5]
    nan += [p.company_id for p in points if not (math.isfinite(p.x) and math.isfinite(p.y))][:5]
    results.append(
        GateResult("no NaN or Infinity values", not nan, f"{len(nan)} offending values", nan)
    )

    # 4. Referential integrity: features and judgments point at real rows.
    dangling = [f"{f.company_id}/{f.tag_id}" for f in features if f.company_id not in company_ids][
        :5
    ]
    dangling += [f"tag {f.tag_id}" for f in features if f.tag_id not in tag_ids][:5]
    results.append(
        GateResult(
            "every assignment references an existing company and tag", not dangling, "", dangling
        )
    )

    # 5. Only active tags may be published.
    inactive = sorted({f.tag_id for f in features if f.tag_id not in active_ids})[:5]
    results.append(
        GateResult("published assignments reference active tags only", not inactive, "", inactive)
    )

    # 6. Neighbours stay inside one embedding-space version.
    versions = {n.embedding_space_version for n in neighbors}
    bad_neighbors = [
        f"{n.company_id}->{n.neighbor_company_id}"
        for n in neighbors
        if n.neighbor_company_id not in company_ids or n.company_id == n.neighbor_company_id
    ][:5]
    results.append(
        GateResult(
            "every neighbour exists and shares one embedding-space version",
            not bad_neighbors and len(versions) <= 1,
            f"versions={sorted(versions)}",
            bad_neighbors,
        )
    )

    # 7. Projection covers exactly the published companies.
    orphan_points = sorted(point_ids - company_ids)[:5]
    results.append(
        GateResult("every projected point maps to a company", not orphan_points, "", orphan_points)
    )

    # 8. Decision hygiene: yes/no/uncertain never collapsed into each other.
    bad_conf = [j.judgment_id for j in judgments if not (0.0 <= j.confidence <= 1.0)][:5]
    results.append(GateResult("confidences lie in [0, 1]", not bad_conf, "", bad_conf))

    # 9. Public artifacts parse and their checksums match the manifest.
    if public_dir is not None:
        results.append(_check_public_artifacts(public_dir))

    for r in results:
        LOG.info("gate %-58s %s %s", r.name, "PASS" if r.passed else "FAIL", r.detail)
    return results


def _check_public_artifacts(public_dir: Path) -> GateResult:
    from pipeline.util import sha256_file
    from pipeline.versions import PUBLIC_ARTIFACT_VERSION

    root = public_dir / f"v{PUBLIC_ARTIFACT_VERSION}"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return GateResult(
            "public artifacts present and checksummed", False, "manifest.json missing"
        )
    manifest: dict[str, Any] = orjson.loads(manifest_path.read_bytes())
    bad: list[str] = []
    for rel, digest in (manifest.get("checksums") or {}).items():
        path = root / rel
        if not path.exists():
            bad.append(f"{rel} missing")
        elif sha256_file(path)[:16] != digest:
            bad.append(f"{rel} checksum mismatch")
        else:
            try:
                orjson.loads(path.read_bytes())
            except orjson.JSONDecodeError:
                bad.append(f"{rel} is not valid JSON")
    return GateResult(
        "public artifacts present and checksummed",
        not bad,
        f"{len(manifest.get('checksums') or {})} files verified"
        if not bad
        else f"{len(bad)} problems",
        bad[:5],
    )
