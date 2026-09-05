from pipeline.ontology.discovery import discover_candidates
from pipeline.ontology.mapping import map_source_taxonomy
from pipeline.ontology.merge import apply_merge, propose_merges
from pipeline.ontology.registry import OntologyRegistry

__all__ = [
    "OntologyRegistry",
    "discover_candidates",
    "propose_merges",
    "apply_merge",
    "map_source_taxonomy",
]
