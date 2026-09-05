/**
 * In-browser keyword search.
 *
 * The index is built at publication time as `search/docs.json` (one projected
 * document per company) and hydrated into MiniSearch on the client. Shipping
 * the projected documents rather than a serialised MiniSearch index keeps the
 * payload roughly a third of the size and lets us change ranking without
 * regenerating the dataset.
 *
 * Ranking is deliberately ordered: exact company-name matches first, then tag
 * matches, then descriptions and metadata. Field boosts alone do not guarantee
 * that, so an exact-name bonus is applied on top of the BM25 score.
 */

import MiniSearch, { type SearchResult } from 'minisearch'
import type { SearchDoc } from './types'

export interface SearchHit {
  id: string
  score: number
  /** Which projected fields matched, for the "why did this match?" affordance. */
  matchedFields: string[]
  terms: string[]
}

const FIELD_LABELS: Record<string, string> = {
  n: 'name',
  a: 'former names',
  o: 'one-liner',
  e: 'description',
  g: 'semantic tags',
  x: 'tag aliases',
  d: 'YC industry',
  b: 'batch & location',
}

export function createIndex(docs: SearchDoc[]): MiniSearch<SearchDoc> {
  const index = new MiniSearch<SearchDoc>({
    idField: 'i',
    fields: ['n', 'a', 'o', 'e', 'g', 'x', 'd', 'b'],
    storeFields: ['n'],
    searchOptions: {
      boost: { n: 8, a: 5, g: 3.5, x: 2.5, o: 2, e: 1, d: 1.4, b: 1 },
      prefix: (term) => term.length >= 3,
      // Typo tolerance scales with term length: a 4-letter term tolerates one
      // edit, a 12-letter term two. Applying a flat fuzziness to short terms
      // produces noise.
      fuzzy: (term) => (term.length >= 5 ? 0.2 : term.length >= 4 ? 0.15 : false),
      combineWith: 'AND',
    },
  })
  index.addAll(docs)
  return index
}

/** Split a query into bare terms and quoted phrases. */
export function parseQuery(query: string): { terms: string[]; phrases: string[] } {
  const phrases: string[] = []
  const rest = query.replace(/"([^"]+)"/g, (_, phrase: string) => {
    phrases.push(phrase.trim().toLowerCase())
    return ' '
  })
  const terms = rest.split(/\s+/).map((t) => t.trim()).filter(Boolean)
  return { terms, phrases }
}

export function search(
  index: MiniSearch<SearchDoc>,
  docsById: Map<string, SearchDoc>,
  query: string,
  limit = 200,
): SearchHit[] {
  const trimmed = query.trim()
  if (!trimmed) return []
  const { terms, phrases } = parseQuery(trimmed)

  // A phrase-only query has no bare terms to feed MiniSearch, so search the
  // phrase text and filter for a literal match afterwards.
  const queryText = terms.length > 0 ? terms.join(' ') : phrases.join(' ')
  if (!queryText) return []

  const raw: SearchResult[] = index.search(queryText)
  const normalizedQuery = trimmed.toLowerCase().replace(/"/g, '')

  const hits: SearchHit[] = []
  for (const r of raw) {
    const doc = docsById.get(String(r.id))
    if (!doc) continue

    if (phrases.length > 0) {
      const haystack = `${doc.n} ${doc.a} ${doc.o} ${doc.e} ${doc.g} ${doc.d} ${doc.b}`.toLowerCase()
      if (!phrases.every((p) => haystack.includes(p))) continue
    }

    const name = doc.n.toLowerCase()
    let score = r.score
    // Deterministic tiers on top of BM25, so an exact name always outranks a
    // description that happens to repeat the word many times.
    if (name === normalizedQuery) score += 10_000
    else if (name.startsWith(normalizedQuery)) score += 1_000
    else if (name.includes(normalizedQuery)) score += 200

    hits.push({
      id: String(r.id),
      score,
      matchedFields: Object.keys(r.match ?? {}).length
        ? [...new Set(Object.values(r.match ?? {}).flat())].map((f) => FIELD_LABELS[f] ?? f)
        : [],
      terms: r.terms ?? [],
    })
  }

  hits.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
  return hits.slice(0, limit)
}

/**
 * Build a highlighted snippet around the first matching term.
 * Returns plain text segments, never HTML, so nothing from the dataset can be
 * injected into the DOM as markup.
 */
export function snippet(
  text: string,
  terms: string[],
  radius = 90,
): Array<{ text: string; hit: boolean }> {
  if (!text) return []
  const lower = text.toLowerCase()
  let at = -1
  let matchedTerm = ''
  for (const term of terms) {
    const idx = lower.indexOf(term.toLowerCase())
    if (idx >= 0 && (at < 0 || idx < at)) {
      at = idx
      matchedTerm = term
    }
  }
  if (at < 0) {
    return [{ text: text.slice(0, radius * 2), hit: false }]
  }
  const start = Math.max(0, at - radius)
  const end = Math.min(text.length, at + matchedTerm.length + radius)
  const slice = text.slice(start, end)

  const segments: Array<{ text: string; hit: boolean }> = []
  if (start > 0) segments.push({ text: '…', hit: false })
  const pattern = new RegExp(`(${terms.map(escapeRegExp).filter(Boolean).join('|')})`, 'ig')
  let last = 0
  for (const m of slice.matchAll(pattern)) {
    const idx = m.index ?? 0
    if (idx > last) segments.push({ text: slice.slice(last, idx), hit: false })
    segments.push({ text: m[0], hit: true })
    last = idx + m[0].length
  }
  if (last < slice.length) segments.push({ text: slice.slice(last), hit: false })
  if (end < text.length) segments.push({ text: '…', hit: false })
  return segments
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
