import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDataset } from '../lib/DatasetContext'

/**
 * The global keyword box. It writes straight into the shared filter state, so
 * the map, the search explorer and the tag explorer all narrow together and the
 * URL always describes what is on screen.
 */
export function SearchBox() {
  const { filters, setFilters, searchReady, results, hits } = useDataset()
  const navigate = useNavigate()
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      const typing = target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)
      if (e.key === '/' && !typing) {
        e.preventDefault()
        ref.current?.focus()
      }
      if (e.key === 'Escape' && document.activeElement === ref.current) {
        ref.current?.blur()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <form
      className="searchbox"
      role="search"
      onSubmit={(e) => {
        e.preventDefault()
        navigate({ pathname: '/search', search: window.location.hash.split('?')[1] ?? '' })
      }}
    >
      <label htmlFor="global-search" className="sr-only">
        Search companies, tags, industries and batches
      </label>
      <input
        id="global-search"
        ref={ref}
        type="search"
        className="searchbox__input"
        placeholder={searchReady ? 'Search companies, tags, industries…  (/)' : 'Preparing search…'}
        value={filters.query}
        autoComplete="off"
        spellCheck={false}
        onChange={(e) => setFilters((f) => ({ ...f, query: e.target.value }))}
      />
      {filters.query && (
        <button
          type="button"
          className="searchbox__clear"
          aria-label="Clear search"
          onClick={() => setFilters((f) => ({ ...f, query: '' }))}
        >
          ×
        </button>
      )}
      <p className="sr-only" role="status" aria-live="polite">
        {filters.query
          ? `${hits.length} keyword matches, ${results.length} companies after filters`
          : ''}
      </p>
    </form>
  )
}
