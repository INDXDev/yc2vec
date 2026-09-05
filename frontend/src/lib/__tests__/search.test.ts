import { describe, expect, it } from 'vitest'
import { createIndex, parseQuery, search, snippet } from '../search'
import type { SearchDoc } from '../types'

const docs: SearchDoc[] = [
  {
    i: 'c:1', n: 'Stripe', a: '', o: 'Payments infrastructure for the internet',
    e: 'Stripe builds economic infrastructure for the internet, handling online payments for businesses.',
    g: 'Developer Tooling Enterprise Buyer', x: 'devtools', d: 'B2B Fintech Payments',
    b: 'Summer 2009 San Francisco Active',
  },
  {
    i: 'c:2', n: 'Stripe Analytics', a: '', o: 'Analytics for payment data',
    e: 'A dashboard that visualises payment flows and reconciliation for finance teams.',
    g: 'Structured Data Enterprise Buyer', x: '', d: 'B2B Analytics',
    b: 'Winter 2021 Remote Active',
  },
  {
    i: 'c:3', n: 'Airbnb', a: 'AirBed and Breakfast', o: 'Book homes from local hosts',
    e: 'A marketplace connecting travellers with people renting out their homes.',
    g: 'Marketplace Model Consumer Facing', x: '', d: 'Consumer Travel',
    b: 'Winter 2009 San Francisco Active',
  },
]

const byId = new Map(docs.map((d) => [d.i, d]))
const index = createIndex(docs)

describe('parseQuery', () => {
  it('separates quoted phrases from bare terms', () => {
    expect(parseQuery('payments "economic infrastructure" online')).toEqual({
      terms: ['payments', 'online'],
      phrases: ['economic infrastructure'],
    })
  })

  it('handles an empty query', () => {
    expect(parseQuery('   ')).toEqual({ terms: [], phrases: [] })
  })
})

describe('search ranking', () => {
  it('ranks an exact company-name match first', () => {
    const hits = search(index, byId, 'Stripe')
    expect(hits[0].id).toBe('c:1')
    // The prefix match still appears, just below the exact one.
    expect(hits.map((h) => h.id)).toContain('c:2')
  })

  it('ranks tag matches above description-only matches', () => {
    const hits = search(index, byId, 'Marketplace Model')
    expect(hits[0].id).toBe('c:3')
  })

  it('finds companies through their semantic tags', () => {
    expect(search(index, byId, 'Developer Tooling').map((h) => h.id)).toContain('c:1')
  })

  it('finds companies through tag aliases', () => {
    expect(search(index, byId, 'devtools').map((h) => h.id)).toContain('c:1')
  })

  it('searches metadata fields such as batch and location', () => {
    const hits = search(index, byId, 'Winter 2009')
    expect(hits.map((h) => h.id)).toContain('c:3')
  })

  it('matches former names', () => {
    expect(search(index, byId, 'AirBed').map((h) => h.id)).toContain('c:3')
  })

  it('tolerates a small typo in a long term', () => {
    expect(search(index, byId, 'reconcilation').map((h) => h.id)).toContain('c:2')
  })

  it('requires all terms by default', () => {
    // "payments" and "homes" never co-occur in one document.
    expect(search(index, byId, 'payments homes')).toHaveLength(0)
  })

  it('enforces quoted phrases literally', () => {
    expect(search(index, byId, '"economic infrastructure"').map((h) => h.id)).toEqual(['c:1'])
    expect(search(index, byId, '"infrastructure economic"')).toHaveLength(0)
  })

  it('reports which fields matched', () => {
    const hits = search(index, byId, 'Airbnb')
    expect(hits[0].matchedFields).toContain('name')
  })

  it('returns nothing for an empty query', () => {
    expect(search(index, byId, '   ')).toHaveLength(0)
  })

  it('is deterministic', () => {
    const a = search(index, byId, 'payments').map((h) => h.id)
    const b = search(index, byId, 'payments').map((h) => h.id)
    expect(a).toEqual(b)
  })

  it('respects the result limit', () => {
    expect(search(index, byId, 'Active', 1)).toHaveLength(1)
  })
})

describe('snippet highlighting', () => {
  it('marks matching terms and returns plain text segments', () => {
    const segments = snippet('Stripe builds economic infrastructure for the internet.', ['infrastructure'])
    const hit = segments.find((s) => s.hit)
    expect(hit?.text.toLowerCase()).toBe('infrastructure')
    // Segments are plain strings; nothing is ever injected as HTML.
    expect(segments.every((s) => typeof s.text === 'string')).toBe(true)
    expect(segments.map((s) => s.text).join('')).toContain('economic')
  })

  it('windows long text around the first match', () => {
    const long = 'padding '.repeat(80) + 'NEEDLE' + ' padding'.repeat(80)
    const segments = snippet(long, ['NEEDLE'], 40)
    const text = segments.map((s) => s.text).join('')
    expect(text.length).toBeLessThan(long.length)
    expect(text).toContain('NEEDLE')
    expect(text.startsWith('…')).toBe(true)
  })

  it('falls back to a prefix when nothing matches', () => {
    const segments = snippet('No match in here at all.', ['zzz'])
    expect(segments.every((s) => !s.hit)).toBe(true)
  })

  it('handles regex metacharacters in terms safely', () => {
    expect(() => snippet('a (b) c', ['(b)'])).not.toThrow()
  })

  it('handles empty text', () => {
    expect(snippet('', ['x'])).toEqual([])
  })
})
