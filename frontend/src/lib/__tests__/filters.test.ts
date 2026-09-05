import { describe, expect, it } from 'vitest'
import {
  EMPTY_FILTERS,
  activeFilterCount,
  applyFilters,
  fromSearchParams,
  isEmpty,
  toSearchParams,
  type Filters,
} from '../filters'
import type { Company, Dataset } from '../types'

function company(overrides: Partial<Company> = {}): Company {
  return {
    id: 'c:1', name: 'Acme', oneLiner: 'Widgets', batch: 'Winter 2020', batchYear: 2020,
    status: 'Active', stage: 'Early', industry: 'B2B', subindustry: 'B2B -> Engineering',
    regions: ['United States of America'], location: 'San Francisco', teamSize: 10,
    website: null, ycUrl: null, logoUrl: null, sourceTags: ['SaaS'], topCompany: false,
    isHiring: false, nonprofit: false, tagIds: ['alpha'], tagScores: [0.8], pointIndex: 0,
    ...overrides,
  }
}

const companies: Company[] = [
  company(),
  company({ id: 'c:2', name: 'Beta', batch: 'Summer 2015', batchYear: 2015, industry: 'Consumer',
    subindustry: 'Consumer -> Social', tagIds: ['alpha', 'beta'], tagScores: [0.9, 0.3],
    pointIndex: 1, topCompany: true }),
  company({ id: 'c:3', name: 'Gamma', batch: 'Winter 2024', batchYear: 2024, status: 'Inactive',
    industry: 'Healthcare', subindustry: 'Healthcare -> Diagnostics', regions: ['Europe'],
    tagIds: [], tagScores: [], pointIndex: 2 }),
]

const dataset = {
  points: { ids: companies.map((c) => c.id), cluster: [0, 1, 1], x: [0, 1, 2], y: [0, 1, 2] },
  tagsById: new Map([
    ['alpha', { tag_id: 'alpha', facet: 'industry' }],
    ['beta', { tag_id: 'beta', facet: 'buyer' }],
  ]),
} as unknown as Dataset

const f = (overrides: Partial<Filters> = {}): Filters => ({ ...EMPTY_FILTERS, ...overrides })
const ids = (rows: Company[]) => rows.map((c) => c.id)

describe('URL round-trip', () => {
  it('restores every field exactly', () => {
    const filters = f({
      query: 'payments api', batches: ['Winter 2020', 'Summer 2015'], years: [2010, 2022],
      industries: ['B2B'], tags: ['alpha', 'beta'], tagMode: 'or', facets: ['buyer'],
      statuses: ['Active'], stages: ['Early'], regions: ['Europe'], minConfidence: 0.35,
      topOnly: true, hiringOnly: true, nonprofitOnly: true, clusters: [0, 2],
    })
    expect(fromSearchParams(toSearchParams(filters))).toEqual(filters)
  })

  it('round-trips the empty state to an empty query string', () => {
    expect(toSearchParams(EMPTY_FILTERS).toString()).toBe('')
    expect(fromSearchParams(new URLSearchParams())).toEqual(EMPTY_FILTERS)
  })

  it('preserves extra non-filter params', () => {
    const params = toSearchParams(EMPTY_FILTERS, { selected: 'c:1', color: 'cluster' })
    expect(params.get('selected')).toBe('c:1')
    expect(params.get('color')).toBe('cluster')
  })

  it('ignores a malformed year range rather than throwing', () => {
    expect(fromSearchParams(new URLSearchParams('years=abc')).years).toBeNull()
  })

  it('survives values containing spaces and punctuation', () => {
    const filters = f({ industries: ['B2B -> Engineering, Product and Design'] })
    expect(fromSearchParams(toSearchParams(filters)).industries).toEqual(filters.industries)
  })
})

describe('filter application', () => {
  it('returns everything when empty', () => {
    expect(applyFilters(companies, EMPTY_FILTERS, dataset, null)).toHaveLength(3)
  })

  it('filters by batch', () => {
    expect(ids(applyFilters(companies, f({ batches: ['Winter 2020'] }), dataset, null))).toEqual(['c:1'])
  })

  it('filters by year range inclusively', () => {
    expect(ids(applyFilters(companies, f({ years: [2015, 2020] }), dataset, null))).toEqual(['c:1', 'c:2'])
  })

  it('matches industry or sub-industry', () => {
    expect(ids(applyFilters(companies, f({ industries: ['B2B -> Engineering'] }), dataset, null))).toEqual(['c:1'])
  })

  it('intersects tags in AND mode', () => {
    expect(ids(applyFilters(companies, f({ tags: ['alpha', 'beta'] }), dataset, null))).toEqual(['c:2'])
  })

  it('unions tags in OR mode', () => {
    const rows = applyFilters(companies, f({ tags: ['alpha', 'beta'], tagMode: 'or' }), dataset, null)
    expect(ids(rows)).toEqual(['c:1', 'c:2'])
  })

  it('applies the confidence threshold before the tag filter', () => {
    // c:2 holds "beta" at 0.3, so a 0.5 threshold must exclude it from a beta filter.
    const rows = applyFilters(companies, f({ tags: ['beta'], minConfidence: 0.5 }), dataset, null)
    expect(rows).toHaveLength(0)
  })

  it('filters by tag facet', () => {
    expect(ids(applyFilters(companies, f({ facets: ['buyer'] }), dataset, null))).toEqual(['c:2'])
  })

  it('filters by cluster', () => {
    expect(ids(applyFilters(companies, f({ clusters: [1] }), dataset, null))).toEqual(['c:2', 'c:3'])
  })

  it('filters by region, status and flags', () => {
    expect(ids(applyFilters(companies, f({ regions: ['Europe'] }), dataset, null))).toEqual(['c:3'])
    expect(ids(applyFilters(companies, f({ statuses: ['Inactive'] }), dataset, null))).toEqual(['c:3'])
    expect(ids(applyFilters(companies, f({ topOnly: true }), dataset, null))).toEqual(['c:2'])
  })

  it('composes keyword matches with structural filters', () => {
    const matched = new Set(['c:1', 'c:2'])
    const rows = applyFilters(companies, f({ industries: ['Consumer'] }), dataset, matched)
    expect(ids(rows)).toEqual(['c:2'])
  })

  it('combines several filters conjunctively', () => {
    const rows = applyFilters(
      companies,
      f({ years: [2010, 2022], tags: ['alpha'], statuses: ['Active'] }),
      dataset,
      null,
    )
    expect(ids(rows)).toEqual(['c:1', 'c:2'])
  })

  it('excludes untagged companies from a tag filter', () => {
    expect(ids(applyFilters(companies, f({ tags: ['alpha'] }), dataset, null))).not.toContain('c:3')
  })
})

describe('filter summaries', () => {
  it('detects the empty state', () => {
    expect(isEmpty(EMPTY_FILTERS)).toBe(true)
    expect(isEmpty(f({ query: 'x' }))).toBe(false)
  })

  it('counts active filters', () => {
    expect(activeFilterCount(EMPTY_FILTERS)).toBe(0)
    expect(activeFilterCount(f({ query: 'a', tags: ['x', 'y'], topOnly: true }))).toBe(4)
  })
})
