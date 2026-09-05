"""Dataset quality metrics.

None of these numbers gate a release on their own; they are published so that
readers can judge how much to trust the dataset. Where a metric needs human
labels it reads the gold review set under ``tests/fixtures/gold/``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline.models import CompanyTagFeature, CompanyTagJudgment, Tag
from pipeline.util import log, normalize_name, read_json

LOG = log(__name__)


def evaluate_dataset(
    *,
    companies_count: int,
    tags: list[Tag],
    features: list[CompanyTagFeature],
    judgments: list[CompanyTagJudgment],
    gold_path: Path | None = None,
) -> dict[str, Any]:
    active = [t for t in tags if t.state == "active"]
    prevalence = Counter(f.tag_id for f in features)
    decisions = Counter(j.decision for j in judgments)

    # Orphan tags: activated but never actually assigned to anyone. A high rate
    # means discovery is producing tags the assigner cannot ground.
    orphan = [t.tag_id for t in active if prevalence.get(t.tag_id, 0) == 0]

    # Duplicate rate: distinct active tags whose normalised names collide. This
    # should be zero; anything else means the merge stage let a duplicate through.
    names: dict[str, list[str]] = defaultdict(list)
    for t in active:
        names[normalize_name(t.canonical_name)].append(t.tag_id)
    duplicates = {k: v for k, v in names.items() if len(v) > 1}

    # Evidence coverage over positive judgments.
    positives = [j for j in judgments if j.decision == "yes"]
    with_evidence = sum(1 for j in positives if j.evidence)

    # Hard negatives are the honest precision probe: they were selected to look
    # plausible, so the share the model accepts approximates its false-positive rate.
    hard = [j for j in judgments if j.shortlist_reason == "hard_negative"]
    hard_yes = sum(1 for j in hard if j.decision == "yes")

    metrics: dict[str, Any] = {
        "companies": companies_count,
        "active_tags": len(active),
        "candidate_tags": sum(1 for t in tags if t.state == "candidate"),
        "merged_tags": sum(1 for t in tags if t.state == "merged"),
        "assignments": len(features),
        "assignments_per_company": round(len(features) / max(1, companies_count), 2),
        "judgments": len(judgments),
        "decision_mix": {k: decisions.get(k, 0) for k in ("yes", "no", "uncertain")},
        "uncertain_rate": round(decisions.get("uncertain", 0) / max(1, len(judgments)), 4),
        "evidence_coverage": round(with_evidence / max(1, len(positives)), 4),
        "orphan_tag_rate": round(len(orphan) / max(1, len(active)), 4),
        "orphan_tags": orphan[:20],
        "duplicate_tag_rate": round(len(duplicates) / max(1, len(active)), 4),
        "duplicate_tag_groups": {k: v for k, v in list(duplicates.items())[:10]},
        "hard_negative_acceptance_rate": round(hard_yes / max(1, len(hard)), 4),
        "hard_negatives_judged": len(hard),
        "tag_prevalence_top": [
            {"tag_id": t, "companies": n} for t, n in prevalence.most_common(15)
        ],
        "tag_prevalence_median": _median([prevalence.get(t.tag_id, 0) for t in active]),
        "facet_distribution": dict(Counter(t.facet for t in active).most_common()),
    }

    gold = _load_gold(gold_path)
    if gold:
        metrics["reviewed_sample"] = _score_against_gold(judgments, gold)
    return metrics


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _load_gold(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = read_json(path, default={})
    return list(data.get("judgments") or [])


def _score_against_gold(
    judgments: list[CompanyTagJudgment], gold: list[dict[str, Any]]
) -> dict[str, Any]:
    """Precision/recall of positive assignments on the human-reviewed sample."""
    predicted = {(j.company_id, j.tag_id): j.decision for j in judgments}
    tp = fp = fn = matched = 0
    for row in gold:
        key = (row["company_id"], row["tag_id"])
        if key not in predicted:
            continue
        matched += 1
        want = row["decision"] == "yes"
        got = predicted[key] == "yes"
        tp += want and got
        fp += (not want) and got
        fn += want and (not got)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "gold_pairs": len(gold),
        "matched_pairs": matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
    }
