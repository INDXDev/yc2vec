import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useSearchParams } from 'react-router-dom'
import MiniSearch from 'minisearch'
import { loadDataset, loadSearchDocs } from './data'
import { createIndex, search as runSearch, type SearchHit } from './search'
import {
  EMPTY_FILTERS,
  applyFilters,
  fromSearchParams,
  toSearchParams,
  type Filters,
} from './filters'
import type { Company, Dataset, SearchDoc } from './types'

interface DatasetState {
  dataset: Dataset | null
  error: Error | null
  loading: boolean
  filters: Filters
  setFilters: (next: Filters | ((prev: Filters) => Filters)) => void
  resetFilters: () => void
  /** Companies passing every filter, including the keyword query. */
  results: Company[]
  /** Ranked keyword hits, empty when there is no query. */
  hits: SearchHit[]
  searchDocs: Map<string, SearchDoc>
  searchReady: boolean
}

const Ctx = createContext<DatasetState | null>(null)

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [searchParams, setSearchParams] = useSearchParams()

  const [index, setIndex] = useState<MiniSearch<SearchDoc> | null>(null)
  const [searchDocs, setSearchDocs] = useState<Map<string, SearchDoc>>(new Map())

  useEffect(() => {
    const controller = new AbortController()
    loadDataset(controller.signal).then(setDataset).catch((e: Error) => {
      if (e.name !== 'AbortError') setError(e)
    })
    return () => controller.abort()
  }, [])

  // The search index is a second, larger payload. It loads after the dataset so
  // the map is interactive first; until it arrives, filters still work and the
  // search box reports that it is still preparing.
  useEffect(() => {
    if (!dataset) return
    const controller = new AbortController()
    loadSearchDocs(controller.signal)
      .then((docs) => {
        setSearchDocs(new Map(docs.map((d) => [d.i, d])))
        setIndex(createIndex(docs))
      })
      .catch(() => {
        /* Search is an enhancement; the rest of the site works without it. */
      })
    return () => controller.abort()
  }, [dataset])

  const filters = useMemo(() => fromSearchParams(searchParams), [searchParams])

  const setFilters = useCallback(
    (next: Filters | ((prev: Filters) => Filters)) => {
      setSearchParams(
        (prev) => {
          const current = fromSearchParams(prev)
          const value = typeof next === 'function' ? next(current) : next
          // Non-filter params (the selected company, the active tab) survive a
          // filter change so the URL stays a complete description of the view.
          const preserved: Record<string, string> = {}
          for (const key of ['selected', 'space', 'color']) {
            const v = prev.get(key)
            if (v) preserved[key] = v
          }
          return toSearchParams(value, preserved)
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const resetFilters = useCallback(() => setFilters(EMPTY_FILTERS), [setFilters])

  // Debounce the keyword query: typing should not re-rank thousands of rows on
  // every keystroke.
  const [debouncedQuery, setDebouncedQuery] = useState(filters.query)
  const timer = useRef<number>()
  useEffect(() => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setDebouncedQuery(filters.query), 160)
    return () => window.clearTimeout(timer.current)
  }, [filters.query])

  const hits = useMemo(() => {
    if (!index || !debouncedQuery.trim()) return []
    return runSearch(index, searchDocs, debouncedQuery, 500)
  }, [index, searchDocs, debouncedQuery])

  const results = useMemo(() => {
    if (!dataset) return []
    const matchIds = debouncedQuery.trim() && index ? new Set(hits.map((h) => h.id)) : null
    const filtered = applyFilters(dataset.companies, filters, dataset, matchIds)
    if (!matchIds) return filtered
    // Preserve relevance order when a query is active.
    const rank = new Map(hits.map((h, i) => [h.id, i]))
    return [...filtered].sort((a, b) => (rank.get(a.id) ?? 0) - (rank.get(b.id) ?? 0))
  }, [dataset, filters, hits, index, debouncedQuery])

  const value: DatasetState = {
    dataset,
    error,
    loading: !dataset && !error,
    filters,
    setFilters,
    resetFilters,
    results,
    hits,
    searchDocs,
    searchReady: index !== null,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useDataset(): DatasetState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDataset must be used inside a DatasetProvider')
  return ctx
}
