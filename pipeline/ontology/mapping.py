"""Relate YC/yc-oss source taxonomy terms to YC2Vec semantic tags.

This mapping is a *comparison artifact*, not a rename. Source terms keep their
exact names and ids; the mapping table records how each YC2Vec tag aligns with,
refines or cuts across them, which is what lets the site answer "what does the
inferred ontology add beyond YC's own categories?".
"""

from __future__ import annotations

import numpy as np

from pipeline.models import SourceTaxonomyTagMapping, SourceTaxonomyTerm, Tag
from pipeline.ollama import OllamaClient
from pipeline.ontology.merge import embed_tags
from pipeline.util import log, normalize_name, now, stable_hash

LOG = log(__name__)

#: Relation thresholds on cosine similarity between a term and a tag definition.
EQUIVALENT = 0.90
OVERLAPS = 0.78
RELATED = 0.68


async def map_source_taxonomy(
    *,
    client: OllamaClient,
    terms: list[SourceTaxonomyTerm],
    tags: list[Tag],
    top_k: int = 3,
    batch_size: int = 16,
) -> list[SourceTaxonomyTagMapping]:
    if not terms or not tags:
        return []

    docs = [f"{t.name} ({t.kind}): a Y Combinator classification for companies." for t in terms]
    vectors: list[list[float]] = []
    for i in range(0, len(docs), batch_size):
        vectors.extend(await client.embed(docs[i : i + batch_size]))
    term_matrix = np.asarray(vectors, dtype=np.float32)
    term_matrix /= np.clip(np.linalg.norm(term_matrix, axis=1, keepdims=True), 1e-12, None)

    tag_matrix = await embed_tags(client, tags, batch_size)
    sims = term_matrix @ tag_matrix.T

    mappings: list[SourceTaxonomyTagMapping] = []
    for i, term in enumerate(terms):
        order = np.argsort(-sims[i])[:top_k]
        for j in order:
            sim = float(sims[i, j])
            tag = tags[int(j)]
            relation = _relation(term, tag, sim)
            if relation is None:
                continue
            mappings.append(
                SourceTaxonomyTagMapping(
                    mapping_id=stable_hash({"t": term.term_id, "g": tag.tag_id})[:16],
                    term_id=term.term_id,
                    tag_id=tag.tag_id,
                    relation=relation,  # type: ignore[arg-type]
                    similarity=sim,
                    method="embedding",
                    reviewed=False,
                    created_at=now(),
                )
            )
    LOG.info("source taxonomy mapping: %d relationships over %d terms", len(mappings), len(terms))
    return mappings


def _relation(term: SourceTaxonomyTerm, tag: Tag, sim: float) -> str | None:
    if normalize_name(term.name) == normalize_name(tag.canonical_name):
        return "equivalent"
    if sim >= EQUIVALENT:
        return "equivalent"
    if sim >= OVERLAPS:
        # A YC industry is a broad bucket; a semantic tag that matches it
        # closely is usually a refinement of it rather than its equal.
        return "narrower" if term.kind in ("industry", "batch") else "overlaps"
    if sim >= RELATED:
        return "related"
    return None
