/**
 * Filter state, and its serialisation into the URL.
 *
 * Every view reads from one filter object, and that object round-trips through
 * the query string, so any view the user is looking at is shareable. Keyword
 * search is part of the same state rather than a separate mode, which is what
 * makes "search + filter + map selection" compose predictably.
 */

import type { Company, Dataset } from './types'

export interface Filters {
  query: string
  batches: string[]
  years: [number, number] | null
  industries: string[]
  /** Semantic tag ids. `tagMode` decides whether they intersect or union. */
  tags: string[]
  tagMode: 'and' | 'or'
  facets: string[]
  statuses: string[]
  stages: string[]
  regions: string[]
  minConfidence: number
  topOnly: boolean
  hiringOnly: boolean
  nonprofitOnly: boolean
  clusters: number[]
}

export const EMPTY_FILTERS: Filters = {
  query: '',
  batches: [],
  years: null,
  industries: [],
  tags: [],
  tagMode: 'and',
  facets: [],
  statuses: [],
  stages: [],
  regions: [],
  minConfidence: 0,
  topOnly: false,
  hiringOnly: false,
  nonprofitOnly: false,
  clusters: [],
}

export function isEmpty(f: Filters): boolean {
  return (
    !f.query &&
    f.batches.length === 0 &&
    f.years === null &&
    f.industries.length === 0 &&
    f.tags.length === 0 &&
    f.facets.length === 0 &&
    f.statuses.length === 0 &&
    f.stages.length === 0 &&
    f.regions.length === 0 &&
    f.minConfidence === 0 &&
    !f.topOnly &&
    !f.hiringOnly &&
    !f.nonprofitOnly &&
    f.clusters.length === 0
  )
}

export function activeFilterCount(f: Filters): number {
  return (
    (f.query ? 1 : 0) +
    f.batches.length +
    (f.years ? 1 : 0) +
    f.industries.length +
    f.tags.length +
    f.facets.length +
    f.statuses.length +
    f.stages.length +
    f.regions.length +
    (f.minConfidence > 0 ? 1 : 0) +
    (f.topOnly ? 1 : 0) +
    (f.hiringOnly ? 1 : 0) +
    (f.nonprofitOnly ? 1 : 0) +
    f.clusters.length
  )
}

const LIST_KEYS = [
  ['batch', 'batches'],
  ['industry', 'industries'],
  ['tag', 'tags'],
  ['facet', 'facets'],
  ['status', 'statuses'],
  ['stage', 'stages'],
  ['region', 'regions'],
] as const

export function toSearchParams(f: Filters, extra?: Record<string, string>): URLSearchParams {
  const p = new URLSearchParams()
  if (f.query) p.set('q', f.query)
  for (const [param, key] of LIST_KEYS) {
    const values = f[key]
    if (values.length) p.set(param, values.join('~'))
  }
  if (f.years) p.set('years', `${f.years[0]}-${f.years[1]}`)
  if (f.tagMode !== 'and') p.set('mode', f.tagMode)
  if (f.minConfidence > 0) p.set('conf', String(f.minConfidence))
  if (f.topOnly) p.set('top', '1')
  if (f.hiringOnly) p.set('hiring', '1')
  if (f.nonprofitOnly) p.set('nonprofit', '1')
  if (f.clusters.length) p.set('cluster', f.clusters.join('~'))
  for (const [k, v] of Object.entries(extra ?? {})) {
    if (v) p.set(k, v)
  }
  return p
}

export function fromSearchParams(p: URLSearchParams): Filters {
  const list = (key: string): string[] => {
    const raw = p.get(key)
    return raw ? raw.split('~').filter(Boolean) : []
  }
  let years: [number, number] | null = null
  const rawYears = p.get('years')
  if (rawYears) {
    const [a, b] = rawYears.split('-').map((n) => Number.parseInt(n, 10))
    if (Number.isFinite(a) && Number.isFinite(b)) years = [a, b]
  }
  return {
    query: p.get('q') ?? '',
    batches: list('batch'),
    years,
    industries: list('industry'),
    tags: list('tag'),
    tagMode: p.get('mode') === 'or' ? 'or' : 'and',
    facets: list('facet'),
    statuses: list('status'),
    stages: list('stage'),
    regions: list('region'),
    minConfidence: Number.parseFloat(p.get('conf') ?? '0') || 0,
    topOnly: p.get('top') === '1',
    hiringOnly: p.get('hiring') === '1',
    nonprofitOnly: p.get('nonprofit') === '1',
    clusters: list('cluster').map((n) => Number.parseInt(n, 10)).filter(Number.isFinite),
  }
}

/**
 * Apply every non-keyword filter. Keyword results are intersected separately by
 * the caller, because the search index is loaded lazily and a filter pass must
 * still work before it arrives.
 */
export function applyFilters(
  companies: Company[],
  f: Filters,
  dataset: Dataset,
  matchIds: Set<string> | null,
): Company[] {
  const wantTags = new Set(f.tags)
  const wantFacets = new Set(f.facets)
  const wantClusters = new Set(f.clusters)

  return companies.filter((c) => {
    if (matchIds && !matchIds.has(c.id)) return false
    if (f.batches.length && (!c.batch || !f.batches.includes(c.batch))) return false
    if (f.years && (c.batchYear === null || c.batchYear < f.years[0] || c.batchYear > f.years[1]))
      return false
    if (f.industries.length) {
      const hit = f.industries.includes(c.industry ?? '') ||
        (c.subindustry ? f.industries.includes(c.subindustry) : false)
      if (!hit) return false
    }
    if (f.statuses.length && !f.statuses.includes(c.status ?? '')) return false
    if (f.stages.length && !f.stages.includes(c.stage ?? '')) return false
    if (f.regions.length && !c.regions.some((r) => f.regions.includes(r))) return false
    if (f.topOnly && !c.topCompany) return false
    if (f.hiringOnly && !c.isHiring) return false
    if (f.nonprofitOnly && !c.nonprofit) return false
    if (wantClusters.size) {
      const idx = c.pointIndex
      if (idx < 0 || !wantClusters.has(dataset.points.cluster[idx])) return false
    }

    if (wantTags.size || wantFacets.size || f.minConfidence > 0) {
      // Confidence is applied first: a tag below the threshold should not
      // satisfy a tag filter, otherwise the two controls contradict each other.
      const passing: string[] = []
      for (let i = 0; i < c.tagIds.length; i += 1) {
        if ((c.tagScores[i] ?? 0) >= f.minConfidence) passing.push(c.tagIds[i])
      }
      if (f.minConfidence > 0 && passing.length === 0) return false
      if (wantTags.size) {
        const set = new Set(passing)
        const ok =
          f.tagMode === 'and'
            ? [...wantTags].every((t) => set.has(t))
            : [...wantTags].some((t) => set.has(t))
        if (!ok) return false
      }
      if (wantFacets.size) {
        const ok = passing.some((t) => wantFacets.has(dataset.tagsById.get(t)?.facet ?? ''))
        if (!ok) return false
      }
    }
    return true
  })
}

/** Distinct facet values, sorted by frequency, for building the filter UI. */
export function facetCounts(companies: Company[], pick: (c: Company) => string[]): Array<[string, number]> {
  const counts = new Map<string, number>()
  for (const c of companies) {
    for (const v of pick(c)) {
      if (v) counts.set(v, (counts.get(v) ?? 0) + 1)
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
}
