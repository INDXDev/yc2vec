"""Ontology migrations.

Tag ids are minted once and frozen, which is what makes historical assignments
interpretable. The corollary is that everything *else* about a tag — its display
name, its aliases, its facet — has to be changeable in place, and changing it
must be a recorded, reversible operation rather than an ad-hoc edit.

Each migration here is idempotent: running it twice is the same as running it
once, so it is safe to include in a scheduled pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.ontology.discovery import _display_name
from pipeline.ontology.registry import OntologyRegistry
from pipeline.util import log, normalize_name, now

LOG = log(__name__)


@dataclass
class MigrationResult:
    name: str
    changed: int
    examples: list[tuple[str, str]]


def normalize_display_names(
    registry: OntologyRegistry, *, dry_run: bool = False
) -> MigrationResult:
    """Re-apply display-name formatting to every tag, leaving ids untouched.

    Models emit the same attribute as ``ai_agents``, ``Ai Agents`` and
    ``AI-Agents`` across runs, and the formatting rules improve over time. The
    old name is kept as an alias so a lookup by the previous spelling still
    resolves — the point of the exercise is that nothing downstream breaks.
    """
    changed: list[tuple[str, str]] = []
    for tag in sorted(registry.tags.values(), key=lambda t: t.tag_id):
        better = _display_name(tag.canonical_name)
        if not better or better == tag.canonical_name:
            continue
        changed.append((tag.canonical_name, better))
        if dry_run:
            continue
        previous = tag.canonical_name
        tag.canonical_name = better
        tag.updated_at = now()
        # Keep the old spelling resolvable.
        if normalize_name(previous) != normalize_name(better):
            registry.add_alias(tag.tag_id, previous, origin="manual")
        elif normalize_name(better) not in tag.normalized_aliases:
            tag.normalized_aliases = sorted({*tag.normalized_aliases, normalize_name(better)})

    if not dry_run and changed:
        registry._reindex()
    LOG.info("normalize_display_names: %d tag(s) renamed", len(changed))
    return MigrationResult("normalize_display_names", len(changed), changed[:10])


MIGRATIONS = {
    "normalize-display-names": normalize_display_names,
}
