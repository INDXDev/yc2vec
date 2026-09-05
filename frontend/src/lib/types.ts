/**
 * Types mirroring the published artifacts in `data/public/v1/`.
 *
 * These correspond to the JSON Schemas emitted by `uv run yc2vec schemas`;
 * `npm run typecheck` is what catches a pipeline schema change that the UI has
 * not been updated for.
 */

export interface ReleaseManifest {
  dataset_version: string
  schema_version: string
  public_artifact_version: string
  pipeline_version: string
  ontology_version: string
  embedding_space_version: string
  projection_version: string
  generated_at: string
  git_commit: string | null
  source_retrieved_at: string | null
  source_last_updated: string | null
  source_url: string
  models: Record<string, unknown>
  prompt_hashes: Record<string, string>
  counts: Record<string, number>
  checksums: Record<string, string>
  limitations: string[]
  licenses: Array<Record<string, string>>
  attribution: string
  detail_shards: number
  key_map: Record<string, string>
}

/** Parallel arrays: the first payload the map needs, and the smallest. */
export interface PointsPayload {
  version: string
  embedding_space_version: string
  count: number
  ids: string[]
  x: number[]
  y: number[]
  cluster: number[]
  year: number[]
  note: string
}

/** A row of `companies.json`, after the short transport keys are expanded. */
export interface Company {
  id: string
  name: string
  oneLiner: string
  batch: string | null
  batchYear: number | null
  status: string | null
  stage: string | null
  industry: string | null
  subindustry: string | null
  regions: string[]
  location: string | null
  teamSize: number | null
  website: string | null
  ycUrl: string | null
  logoUrl: string | null
  sourceTags: string[]
  topCompany: boolean
  isHiring: boolean
  nonprofit: boolean
  tagIds: string[]
  tagScores: number[]
  /** Index into the points arrays; -1 when the company has no projection. */
  pointIndex: number
}

export interface Tag {
  tag_id: string
  name: string
  facet: string
  definition: string
  aliases: string[]
  parents: string[]
  prevalence: number
  support: number
  examples: string[]
  cooccurring: Array<{ tag_id: string; count: number }>
  by_year: Record<string, number>
}

export interface TagsPayload {
  ontology_version: string
  count: number
  facets: string[]
  rows: Tag[]
}

export interface TaxonomyTerm {
  term_id: string
  kind: string
  name: string
  parent: string | null
  count: number
}

export interface TaxonomyMapping {
  term_id: string
  tag_id: string
  relation: string
  similarity: number
  reviewed: boolean
}

export interface TaxonomyPayload {
  terms: TaxonomyTerm[]
  mappings: TaxonomyMapping[]
}

export interface ClusterRow {
  cluster_id: number
  label: string
  size: number
  top_tag_ids: string[]
  x: number
  y: number
}

export interface ClustersPayload {
  projection_version: string
  disclaimer: string
  rows: ClusterRow[]
}

export interface AssignedTag {
  tag_id: string
  name: string
  facet: string
  value: number
  confidence: number
  raw_confidence: number
  weight: number
  rationale: string | null
  evidence: Array<{ doc: string; quote: string }>
  shortlist_reason: string | null
}

export interface NeighborEntry {
  id: string
  name: string
  one_liner: string
  batch: string | null
  score: number
  shared_tags?: string[]
  shared_metadata?: string[]
}

/** Every similarity mode the detail view can switch between. */
export type SimilaritySpace = 'combined' | 'description' | 'metadata' | 'tags' | 'sparse_tags'

export const SIMILARITY_SPACES: Array<{ key: SimilaritySpace; label: string; blurb: string }> = [
  {
    key: 'combined',
    label: 'Combined',
    blurb: 'Description, metadata and assigned tags, weighted and renormalised.',
  },
  {
    key: 'description',
    label: 'Description',
    blurb: 'The company’s own description text only.',
  },
  {
    key: 'metadata',
    label: 'Metadata',
    blurb: 'Batch, industry, region, stage and other structured fields only.',
  },
  {
    key: 'tags',
    label: 'Semantic tags',
    blurb: 'Dense embedding of the assigned tag definitions.',
  },
  {
    key: 'sparse_tags',
    label: 'Sparse tag profile',
    blurb: 'Overlap of the interpretable tag vectors (weighted Jaccard).',
  },
]

export interface CompanyDetail {
  company_id: string
  name: string
  one_liner: string | null
  long_description: string | null
  website: string | null
  yc_url: string
  logo_url: string | null
  batch: string | null
  batch_year: number | null
  status: string | null
  stage: string | null
  team_size: number | null
  industry: string | null
  subindustry: string | null
  industries: string[]
  source_tags: string[]
  regions: string[]
  all_locations: string | null
  former_names: string[]
  top_company: boolean
  is_hiring: boolean
  nonprofit: boolean
  metadata_document: string
  source_taxonomy_term_ids: string[]
  tags: AssignedTag[]
  uncertain_tags: Array<{ tag_id: string; name: string; notes: string | null }>
  neighbors: Partial<Record<SimilaritySpace, NeighborEntry[]>>
  coordinates: { x: number; y: number; cluster: number } | null
}

export interface SearchDoc {
  i: string
  n: string
  a: string
  o: string
  e: string
  g: string
  x: string
  d: string
  b: string
}

export interface Dataset {
  manifest: ReleaseManifest
  points: PointsPayload
  companies: Company[]
  companiesById: Map<string, Company>
  tags: TagsPayload
  tagsById: Map<string, Tag>
  taxonomy: TaxonomyPayload
  clusters: ClustersPayload
}
