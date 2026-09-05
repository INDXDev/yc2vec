"""A deterministic stand-in for Ollama, used by the ``fixture`` profile.

CI must be able to exercise the whole vertical slice — discovery, merge review,
assignment, embeddings, neighbours, projection, publication and the gates —
without downloading a 18 GB model. This module provides a model-shaped backend
that is:

* **Deterministic.** Every response is a pure function of the request, so the
  fixture dataset is byte-reproducible and a diff in CI means a real change.
* **Schema-valid.** Output is generated to satisfy the same JSON Schemas the
  real client validates against, so the parsing and repair paths downstream are
  genuinely exercised.
* **Honest.** The vocabulary comes from a committed file of curated responses,
  and the embeddings are hashed pseudo-vectors. Nothing here is a claim about
  what a real model would say; it is a harness.

The synthetic embedding deserves a note: it hashes character n-grams into a
fixed-width vector, so texts sharing vocabulary land near each other. That is
enough structure for neighbours, UMAP and clustering to produce a non-degenerate
fixture map, which is what the UI tests need.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from pipeline.util import read_json, stable_hash

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
RESPONSES_PATH = FIXTURE_DIR / "model_responses.json"

#: Dimension of the synthetic embedding. Small enough to keep the fixture
#: artifacts tiny, large enough for hashed n-grams not to collide constantly.
FIXTURE_DIM = 128

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def fixture_embedding(text: str, dim: int = FIXTURE_DIM) -> list[float]:
    """A hashed bag-of-n-grams vector: deterministic, and semantically ordered
    enough that similar texts are actually near each other."""
    vector = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        for gram in {token, *(token[i : i + 4] for i in range(max(1, len(token) - 3)))}:
            h = _seed(gram)
            vector[h % dim] += 1.0 if h >> 8 & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-9:
        # An empty or unhashable document still needs a valid unit vector.
        return [1.0 if i == 0 else 0.0 for i in range(dim)]
    return [v / norm for v in vector]


class FixtureBackend:
    """Serves canned responses, falling back to deterministic synthesis."""

    def __init__(self, responses_path: Path = RESPONSES_PATH) -> None:
        self.canned: dict[str, Any] = read_json(responses_path, default={}) or {}

    # -- generation ------------------------------------------------------
    def generate(
        self, *, system: str, prompt: str, schema: dict[str, Any], namespace: str
    ) -> dict[str, Any]:
        key = stable_hash({"system": system, "prompt": prompt, "schema": schema})
        if key in self.canned:
            return self.canned[key]
        if namespace == "discovery":
            return self._discovery(prompt)
        if namespace == "merge":
            return self._merge(prompt)
        if namespace == "assign":
            return self._assign(prompt)
        if namespace == "cluster":
            return {"label": "Mixed cluster"}
        raise KeyError(f"fixture backend has no response for namespace {namespace!r}")

    # A small controlled vocabulary, one entry per facet, so the fixture
    # ontology exercises facet grouping without pretending to be a real one.
    VOCAB: tuple[tuple[str, str, str], ...] = (
        (
            "Developer Tooling",
            "product_form",
            "The product is built for software engineers to use directly in their work.",
        ),
        (
            "Enterprise Buyer",
            "buyer",
            "The purchase decision sits with a company rather than an individual consumer.",
        ),
        (
            "Marketplace Model",
            "business_model",
            "The company connects two distinct sides of a transaction and takes a cut.",
        ),
        (
            "Subscription Revenue",
            "business_model",
            "Revenue is recurring rather than transactional or one-off.",
        ),
        (
            "Workflow Automation",
            "workflow",
            "The product removes manual steps from an existing business process.",
        ),
        (
            "Machine Learning Core",
            "technology",
            "Statistical or learned models are central to how the product works.",
        ),
        (
            "Structured Data",
            "data_modality",
            "The product primarily operates on tabular or record-shaped data.",
        ),
        (
            "Regulated Domain",
            "regulation",
            "Operating in this space requires meeting sector-specific legal obligations.",
        ),
        (
            "Physical Operations",
            "deployment",
            "Delivering the product requires hardware, facilities or field operations.",
        ),
        (
            "Consumer Facing",
            "customer",
            "The end user is an individual person acting for themselves.",
        ),
        (
            "Healthcare Sector",
            "industry",
            "The company serves clinical, payer or patient-facing healthcare needs.",
        ),
        (
            "Financial Services",
            "industry",
            "The company serves banking, payments, lending or investment needs.",
        ),
    )

    def _discovery(self, prompt: str) -> dict[str, Any]:
        ids = re.findall(r"^\[([^\]]+)\]", prompt, re.MULTILINE)
        if not ids:
            ids = ["fixture:1"]
        rng = _seed(prompt)
        candidates = []
        # Deterministically pick a rotating window of the vocabulary so
        # different batches propose overlapping-but-not-identical sets, which is
        # what makes the merge stage do real work.
        start = rng % len(self.VOCAB)
        for offset in range(5):
            name, facet, definition = self.VOCAB[(start + offset) % len(self.VOCAB)]
            support = [ids[(rng + offset + i) % len(ids)] for i in range(min(3, len(ids)))]
            candidates.append(
                {
                    "name": name,
                    "facet": facet,
                    "definition": definition,
                    "positive_examples": [],
                    "negative_examples": [],
                    "supporting_company_ids": sorted(set(support)),
                }
            )
        return {"candidates": candidates}

    def _merge(self, prompt: str) -> dict[str, Any]:
        # Merge only when the two definitions are literally the same text; the
        # fixture must never invent a merge the data does not support.
        definitions = re.findall(r"definition: (.+)", prompt)
        same = len(definitions) == 2 and definitions[0].strip() == definitions[1].strip()
        return {
            "verdict": "merge" if same else "distinct",
            "rationale": "Identical definitions."
            if same
            else "The definitions describe different attributes.",
        }

    def _assign(self, prompt: str) -> dict[str, Any]:
        tag_ids = re.findall(r"tag_id: ([\w:.-]+)", prompt)
        doc_ids = re.findall(r"document_id: ([^\s(]+)", prompt)
        # Quote a real span from the evidence so the verification step passes
        # for positives and fails for nothing by accident.
        evidence_text = ""
        block = re.search(r"<untrusted_evidence>\s*(.+?)\s*</untrusted_evidence>", prompt, re.S)
        if block:
            evidence_text = " ".join(block.group(1).split())

        judgments = []
        for tag_id in dict.fromkeys(tag_ids):
            h = _seed(prompt[:200] + tag_id)
            bucket = h % 10
            if bucket < 4 and evidence_text and doc_ids:
                words = evidence_text.split()
                quote = " ".join(words[: min(12, len(words))])
                judgments.append(
                    {
                        "tag_id": tag_id,
                        "decision": "yes",
                        "confidence": round(0.6 + (h % 35) / 100, 2),
                        "rationale": "The supplied description matches this attribute's definition.",
                        "evidence": [{"document_id": doc_ids[0], "quote": quote}],
                    }
                )
            elif bucket < 8:
                judgments.append(
                    {
                        "tag_id": tag_id,
                        "decision": "no",
                        "confidence": round(0.5 + (h % 40) / 100, 2),
                        "rationale": "The evidence does not indicate this attribute.",
                        "evidence": [],
                    }
                )
            else:
                judgments.append(
                    {
                        "tag_id": tag_id,
                        "decision": "uncertain",
                        "confidence": round((h % 40) / 100, 2),
                        "rationale": "The evidence is too thin to decide.",
                        "evidence": [],
                        "notes": "insufficient public description",
                    }
                )
        return {"judgments": judgments}

    # -- embeddings ------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [fixture_embedding(t) for t in texts]
