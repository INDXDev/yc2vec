"""Version constants that participate in cache keys and provenance records.

Every derived artifact records the versions that produced it so that a stale
record can always be traced back to the exact code path, prompt and model.
Bump the relevant constant whenever the semantics of a stage change; the
manifest/DAG in :mod:`pipeline.store` then reruns only the affected stages.
"""

from __future__ import annotations

# Schema version for every table written under data/. Bump on breaking changes.
SCHEMA_VERSION = "1"

# Normalisation logic version (field cleanup, ID minting, canonical documents).
NORMALIZE_VERSION = "1"

# Website main-text extraction logic version.
EXTRACTION_VERSION = "1"

# Deterministic metadata-document template version.
METADATA_TEMPLATE_VERSION = "1"

# Deterministic combined-document / vector composition version.
COMBINED_TEMPLATE_VERSION = "1"

# Ontology structure version (facets, lifecycle rules, id minting).
ONTOLOGY_VERSION = "1"

# Public artifact (browser payload) format version.
PUBLIC_ARTIFACT_VERSION = "1"
