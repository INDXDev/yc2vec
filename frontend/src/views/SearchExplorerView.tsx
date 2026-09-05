import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useDataset } from '../lib/DatasetContext'
import { snippet } from '../lib/search'
import '../styles/search.css'

const PAGE = 50

/**
 * A ranked, keyboard-navigable table of companies.
 *
 * This exists so the dataset stays usable without the 2D projection: everything
 * the map can express as a filter is expressible here as a sortable list, and
 * every row links onward to the profile, the map, or similar companies.
 */
export function SearchExplorerView() {
  const { dataset, results, hits, filters, setFilters, searchReady, searchDocs } = useDataset()
  const [params] = useSearchParams()
  const [limit, setLimit] = useState(PAGE)
  const [cursor, setCursor] = useState(0)
  const listRef = useRef<HTMLTableSectionElement>(null)

  // A lasso selection on the map hands over a set of ids; showing them here
  // keeps the spatial gesture honest — it is a set of companies, not a category.
  const lassoIds = useMemo(() => {
    const raw = params.get('ids')
    return raw ? new Set(raw.split('~').filter(Boolean)) : null
  }, [params])

  const rows = useMemo(
    () => (lassoIds ? results.filter((c) => lassoIds.has(c.id)) : results),
    [results, lassoIds],
  )
  const shown = rows.slice(0, limit)

  const hitById = useMemo(() => new Map(hits.map((h) => [h.id, h])), [hits])

  useEffect(() => {
    setLimit(PAGE)
    setCursor(0)
  }, [filters.query, rows.length])

  // Roving focus through the result rows, so the table is usable without a mouse.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault()
        setCursor((c) => {
          const next = Math.min(shown.length - 1, Math.max(0, c + (e.key === 'ArrowDown' ? 1 : -1)))
          listRef.current
            ?.querySelectorAll('tr')
            [next]?.scrollIntoView({ block: 'nearest' })
          return next
        })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [shown.length])

  if (!dataset) {
    return <div className="searchview searchview--message muted">Loading…</div>
  }

  return (
    <div className="searchview scroll">
      <div className="searchview__inner">
        <header className="searchview__header">
          <h1 className="searchview__title">
            {lassoIds ? 'Map selection' : filters.query ? 'Search results' : 'All companies'}
          </h1>
          <p className="muted searchview__summary" role="status" aria-live="polite">
            {rows.length.toLocaleString()} companies
            {filters.query && ` matching “${filters.query}”`}
            {!searchReady && filters.query && ' — search index still loading'}
          </p>
          {lassoIds && (
            <Link className="btn" to="/search">
              Clear map selection
            </Link>
          )}
        </header>

        {rows.length === 0 ? (
          <div className="searchview__empty panel">
            <h2>No companies match</h2>
            <ul className="muted">
              <li>Check the spelling, or try a shorter query — search tolerates small typos only.</li>
              <li>Quoted phrases must match exactly; try removing the quotes.</li>
              <li>
                Tag filters default to matching <strong>ALL</strong> selected tags. Switch to{' '}
                <strong>ANY</strong> in the sidebar to widen the result.
              </li>
            </ul>
            <button className="btn" onClick={() => setFilters((f) => ({ ...f, query: '' }))}>
              Clear the keyword query
            </button>
          </div>
        ) : (
          <>
            <table className="data searchview__table">
              <thead>
                <tr>
                  <th scope="col">Company</th>
                  <th scope="col">Batch</th>
                  <th scope="col">YC industry</th>
                  <th scope="col">Semantic tags</th>
                  <th scope="col">Links</th>
                </tr>
              </thead>
              <tbody ref={listRef}>
                {shown.map((c, i) => {
                  const hit = hitById.get(c.id)
                  const doc = searchDocs.get(c.id)
                  const terms = hit?.terms ?? []
                  const segments =
                    hit && doc
                      ? snippet(doc.e || doc.o || '', terms)
                      : [{ text: c.oneLiner, hit: false }]
                  return (
                    <tr key={c.id} className={i === cursor ? 'is-cursor' : undefined}>
                      <td>
                        <Link className="searchview__name" to={`/company/${encodeURIComponent(c.id)}`}>
                          {c.name}
                        </Link>
                        <p className="searchview__snippet">
                          {segments.map((s, k) =>
                            s.hit ? <mark key={k}>{s.text}</mark> : <span key={k}>{s.text}</span>,
                          )}
                        </p>
                        {hit && hit.matchedFields.length > 0 && (
                          <p className="faint searchview__matched">
                            matched in {hit.matchedFields.join(', ')}
                          </p>
                        )}
                      </td>
                      <td className="searchview__cell">
                        {c.batch}
                        {c.status && <div className="faint">{c.status}</div>}
                      </td>
                      <td className="searchview__cell">
                        {c.industry}
                        {c.location && <div className="faint">{c.location}</div>}
                      </td>
                      <td>
                        <div className="searchview__tags">
                          {c.tagIds.slice(0, 4).map((id) => {
                            const tag = dataset.tagsById.get(id)
                            if (!tag) return null
                            return (
                              <Link key={id} className="pill pill--button" to={`/tags/${encodeURIComponent(id)}`}>
                                {tag.name}
                              </Link>
                            )
                          })}
                          {c.tagIds.length > 4 && (
                            <span className="faint mono">+{c.tagIds.length - 4}</span>
                          )}
                          {c.tagIds.length === 0 && <span className="faint">—</span>}
                        </div>
                      </td>
                      <td className="searchview__links">
                        <Link to={`/?selected=${encodeURIComponent(c.id)}`}>map</Link>
                        <Link to={`/company/${encodeURIComponent(c.id)}?space=combined`}>similar</Link>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {limit < rows.length && (
              <button className="btn searchview__more" onClick={() => setLimit((l) => l + PAGE * 4)}>
                Show more ({(rows.length - limit).toLocaleString()} remaining)
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}
