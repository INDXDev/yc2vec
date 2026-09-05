"""The tag registry: stable ids, aliases, lifecycle and migrations.

Invariants this class enforces:

* A ``tag_id`` is minted once from the name at creation and then frozen. Renaming
  a tag never changes its id, so historical assignments stay interpretable.
* Raw LLM strings never become ids. They enter as candidates, are normalised,
  and are matched against existing aliases before a new tag is minted.
* A merge is a recorded migration (``state='merged'``, ``merged_into=...``),
  not a deletion, so old judgments can still be resolved to a live tag.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from pipeline.models import Tag, TagAlias, TagCandidate
from pipeline.util import log, normalize_name, now, read_jsonl, slugify, write_jsonl
from pipeline.versions import ONTOLOGY_VERSION

LOG = log(__name__)


class OntologyRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tags_path = self.root / "tags.jsonl"
        self.aliases_path = self.root / "tag_aliases.jsonl"
        self.candidates_path = self.root / "tag_candidates.jsonl"
        self.tags: dict[str, Tag] = {}
        self.aliases: dict[str, TagAlias] = {}
        self.candidates: dict[str, TagCandidate] = {}
        #: normalized alias/name -> tag_id, the dedup index.
        self._index: dict[str, str] = {}
        self.load()

    # -- persistence ------------------------------------------------------
    def load(self) -> None:
        self.tags = {r["tag_id"]: Tag(**r) for r in read_jsonl(self.tags_path)}
        self.aliases = {r["alias_id"]: TagAlias(**r) for r in read_jsonl(self.aliases_path)}
        self.candidates = {
            r["candidate_id"]: TagCandidate(**r) for r in read_jsonl(self.candidates_path)
        }
        self._reindex()

    def save(self) -> None:
        write_jsonl(self.tags_path, sorted(self.tags.values(), key=lambda t: t.tag_id))
        write_jsonl(self.aliases_path, sorted(self.aliases.values(), key=lambda a: a.alias_id))
        write_jsonl(
            self.candidates_path, sorted(self.candidates.values(), key=lambda c: c.candidate_id)
        )

    def _reindex(self) -> None:
        self._index = {}
        for tag in self.tags.values():
            if tag.state == "deprecated":
                continue
            target = tag.merged_into or tag.tag_id
            self._index[normalize_name(tag.canonical_name)] = target
            for normalized in tag.normalized_aliases:
                self._index.setdefault(normalized, target)
        for alias in self.aliases.values():
            self._index.setdefault(alias.normalized_alias, alias.tag_id)

    # -- lookup -----------------------------------------------------------
    def resolve(self, name: str) -> Tag | None:
        """Find a live tag by name or alias, following merges."""
        tag_id = self._index.get(normalize_name(name))
        return self.follow(tag_id) if tag_id else None

    def follow(self, tag_id: str | None) -> Tag | None:
        """Resolve a possibly-merged id to the tag that superseded it."""
        seen: set[str] = set()
        while tag_id and tag_id not in seen:
            seen.add(tag_id)
            tag = self.tags.get(tag_id)
            if tag is None:
                return None
            if tag.state == "merged" and tag.merged_into:
                tag_id = tag.merged_into
                continue
            return tag
        return None

    def active(self) -> list[Tag]:
        return sorted(
            (t for t in self.tags.values() if t.state == "active"), key=lambda t: t.tag_id
        )

    def by_facet(self, facet: str) -> list[Tag]:
        return [t for t in self.active() if t.facet == facet]

    def mint_id(self, name: str, facet: str) -> str:
        """Stable slug id, disambiguated by facet then by counter on collision."""
        base = slugify(name)
        if base not in self.tags:
            return base
        scoped = f"{slugify(facet)}-{base}"
        if scoped not in self.tags:
            return scoped
        i = 2
        while f"{base}-{i}" in self.tags:
            i += 1
        return f"{base}-{i}"

    # -- mutation ---------------------------------------------------------
    def add_candidate(self, candidate: TagCandidate) -> TagCandidate:
        existing = self.candidates.get(candidate.candidate_id)
        if existing is not None:
            # Same proposal seen again: accumulate support rather than duplicate.
            merged = sorted(set(existing.support_company_ids) | set(candidate.support_company_ids))
            existing.support_company_ids = merged
            return existing
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def create_tag(
        self,
        *,
        name: str,
        facet: str,
        definition: str,
        aliases: Iterable[str] = (),
        positive_examples: Iterable[str] = (),
        negative_examples: Iterable[str] = (),
        state: str = "candidate",
        support_company_ids: Iterable[str] = (),
        discovery_run_id: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
        proposer: str = "llm",
        created_at: datetime | None = None,
    ) -> Tag:
        ts = created_at or now()
        alias_list = sorted({a.strip() for a in aliases if a and a.strip() != name})
        tag = Tag(
            tag_id=self.mint_id(name, facet),
            canonical_name=name.strip(),
            definition=definition.strip(),
            facet=facet,
            aliases=alias_list,
            normalized_aliases=sorted({normalize_name(a) for a in [name, *alias_list]}),
            positive_examples=list(positive_examples),
            negative_examples=list(negative_examples),
            state=state,  # type: ignore[arg-type]
            support_count=len(set(support_company_ids)),
            proposer=proposer,
            source_company_ids=sorted(set(support_company_ids)),
            discovery_run_id=discovery_run_id,
            prompt_version=prompt_version,
            model=model,
            ontology_version=ONTOLOGY_VERSION,
            created_at=ts,
            updated_at=ts,
        )
        self.tags[tag.tag_id] = tag
        for alias in tag.normalized_aliases:
            self._index.setdefault(alias, tag.tag_id)
        return tag

    def add_alias(self, tag_id: str, alias: str, origin: str = "merge") -> None:
        norm = normalize_name(alias)
        if not norm:
            return
        alias_id = f"{tag_id}:{slugify(alias)}"
        if alias_id in self.aliases:
            return
        self.aliases[alias_id] = TagAlias(
            alias_id=alias_id,
            tag_id=tag_id,
            alias=alias,
            normalized_alias=norm,
            origin=origin,  # type: ignore[arg-type]
            created_at=now(),
        )
        tag = self.tags.get(tag_id)
        if tag is not None and norm not in tag.normalized_aliases:
            tag.aliases = sorted({*tag.aliases, alias})
            tag.normalized_aliases = sorted({*tag.normalized_aliases, norm})
            tag.updated_at = now()
        self._index.setdefault(norm, tag_id)

    def activate(self, tag_id: str, *, min_support: int, override: bool = False) -> bool:
        """Promote a candidate once it clears support and quality rules."""
        tag = self.tags.get(tag_id)
        if tag is None or tag.state not in ("candidate",):
            return False
        if not override and tag.support_count < min_support:
            return False
        if len(tag.definition) < 20:
            LOG.debug("refusing to activate %s: definition too thin", tag_id)
            return False
        tag.state = "active"
        tag.updated_at = now()
        return True

    def deprecate(self, tag_id: str, reason: str) -> None:
        tag = self.tags.get(tag_id)
        if tag is None:
            return
        tag.state = "deprecated"
        tag.deprecation_reason = reason
        tag.updated_at = now()
        self._reindex()

    def stats(self) -> dict[str, int]:
        counts = {"total": len(self.tags), "candidates_pending": 0}
        for state in ("candidate", "active", "merged", "deprecated"):
            counts[state] = sum(1 for t in self.tags.values() if t.state == state)
        counts["candidates_pending"] = sum(
            1 for c in self.candidates.values() if c.resolution == "pending"
        )
        counts["aliases"] = len(self.aliases)
        return counts
