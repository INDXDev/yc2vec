"""Pick the company/tag pairs worth an LLM call.

Evaluating every company against every tag is quadratic and mostly wasted: the
overwhelming majority of pairs are obvious "no"s. Instead each company gets a
shortlist from four complementary signals, plus calibrated hard negatives so
that precision is measurable rather than assumed:

* **retrieval**      -- cosine similarity between the company's description
                        embedding and each tag's definition embedding;
* **facet_prior**    -- every facet must be represented, so a company is never
                        judged only on whichever facet its description happens
                        to sound like;
* **alias**          -- a literal alias hit in the company's text;
* **metadata_rule**  -- a source taxonomy term that maps to the tag;
* **parent**         -- the parent of any shortlisted tag, so hierarchies stay
                        consistent;
* **hard_negative**  -- plausible-but-probably-wrong tags sampled just below the
                        retrieval cutoff. Their judged precision is the honest
                        estimate of the classifier's false-positive rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.models import CompanyNormalized, SourceTaxonomyTagMapping, Tag
from pipeline.util import normalize_name

ShortlistReason = str


@dataclass(frozen=True)
class ShortlistItem:
    tag_id: str
    reason: ShortlistReason
    score: float


class Shortlister:
    def __init__(
        self,
        tags: list[Tag],
        tag_matrix: np.ndarray,
        *,
        mappings: list[SourceTaxonomyTagMapping] | None = None,
        shortlist_size: int = 24,
        hard_negatives: int = 4,
        seed: int = 20240917,
    ) -> None:
        self.tags = tags
        self.tag_matrix = tag_matrix
        self.tag_index = {t.tag_id: i for i, t in enumerate(tags)}
        self.shortlist_size = shortlist_size
        self.hard_negatives = hard_negatives
        self.rng = np.random.default_rng(seed)
        self.by_facet: dict[str, list[int]] = {}
        for i, t in enumerate(tags):
            self.by_facet.setdefault(t.facet, []).append(i)
        # alias -> tag index, for literal text hits
        self.alias_index: dict[str, int] = {}
        for i, t in enumerate(tags):
            for alias in t.normalized_aliases or [normalize_name(t.canonical_name)]:
                if len(alias) >= 4:
                    self.alias_index.setdefault(alias, i)
        # source term -> tag indices, for metadata rules
        self.term_to_tags: dict[str, list[int]] = {}
        for m in mappings or []:
            if m.relation in ("equivalent", "narrower", "overlaps") and m.tag_id in self.tag_index:
                self.term_to_tags.setdefault(m.term_id, []).append(self.tag_index[m.tag_id])

    def shortlist(
        self, company: CompanyNormalized, company_vector: np.ndarray, company_text: str
    ) -> list[ShortlistItem]:
        if not self.tags:
            return []
        sims = self.tag_matrix @ company_vector
        picked: dict[str, ShortlistItem] = {}

        def add(idx: int, reason: str, score: float) -> None:
            tag_id = self.tags[idx].tag_id
            existing = picked.get(tag_id)
            # First reason wins on ties; a stronger score upgrades the record.
            if existing is None or score > existing.score:
                picked[tag_id] = ShortlistItem(
                    tag_id, existing.reason if existing else reason, score
                )

        # 1. top-N by retrieval
        order = np.argsort(-sims)
        for idx in order[: self.shortlist_size]:
            add(int(idx), "retrieval", float(sims[idx]))

        # 2. best tag in every facet, so no facet is structurally ignored
        for indices in self.by_facet.values():
            best = max(indices, key=lambda i: sims[i])
            add(best, "facet_prior", float(sims[best]))

        # 3. literal alias hits
        text_norm = " " + normalize_name(company_text) + " "
        for alias, idx in self.alias_index.items():
            if f" {alias} " in text_norm:
                add(idx, "alias", float(sims[idx]))

        # 4. metadata rules from the reviewed source-taxonomy mapping
        for term_id in company.source_taxonomy_term_ids:
            for idx in self.term_to_tags.get(term_id, []):
                add(idx, "metadata_rule", float(sims[idx]))

        # 5. parents of shortlisted tags
        for tag_id in list(picked):
            for parent in self.tags[self.tag_index[tag_id]].parent_tag_ids:
                pidx = self.tag_index.get(parent)
                if pidx is not None:
                    add(pidx, "parent", float(sims[pidx]))

        # 6. hard negatives sampled from the band just below the cutoff
        band = [int(i) for i in order[self.shortlist_size : self.shortlist_size * 3]]
        band = [i for i in band if self.tags[i].tag_id not in picked]
        if band and self.hard_negatives:
            chosen = self.rng.choice(
                len(band), size=min(self.hard_negatives, len(band)), replace=False
            )
            for k in np.atleast_1d(chosen):
                idx = band[int(k)]
                picked[self.tags[idx].tag_id] = ShortlistItem(
                    self.tags[idx].tag_id, "hard_negative", float(sims[idx])
                )

        return sorted(picked.values(), key=lambda it: (-it.score, it.tag_id))
