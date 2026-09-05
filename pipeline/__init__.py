"""YC2Vec: an open-data semantic map of Y Combinator companies.

The pipeline is offline-only. It fetches public structured company records,
enriches them with optional website text, discovers a versioned semantic tag
ontology with a fully local LLM, assigns tags with evidence, computes dense
embeddings, nearest neighbours and a 2D UMAP projection, and publishes static
artifacts that a browser can consume without any backend.
"""

__version__ = "0.1.0"

# Bumping PIPELINE_VERSION invalidates every content-addressed cache entry that
# embeds it, which is how we force full recomputation after semantic changes.
PIPELINE_VERSION = "1"
