import { useCallback, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useDataset } from '../lib/DatasetContext'
import { FilterPanel } from '../components/FilterPanel'
import { ScatterCanvas, type ColorBy } from '../components/ScatterCanvas'
import { HoverCard } from '../components/HoverCard'
import { CompanyPreview } from '../components/CompanyPreview'
import '../styles/scatter.css'
import '../styles/filters.css'
import '../styles/map.css'
import '../styles/preview.css'

const COLOR_MODES: Array<{ key: ColorBy; label: string }> = [
  { key: 'year', label: 'Batch year' },
  { key: 'industry', label: 'YC industry' },
  { key: 'cluster', label: 'Cluster' },
  { key: 'tag', label: 'Semantic tag' },
]

export function DnaMapView() {
  const { dataset, results, loading, filters, setFilters } = useDataset()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [fitToken, setFitToken] = useState(0)

  const colorBy = (params.get('color') as ColorBy) || 'year'
  const selectedId = params.get('selected')

  // The scatterplot addresses points by index, so the filtered company list is
  // translated once per change rather than per frame.
  const visible = useMemo(() => {
    const set = new Set<number>()
    for (const c of results) {
      if (c.pointIndex >= 0) set.add(c.pointIndex)
    }
    return set
  }, [results])

  const setParam = useCallback(
    (key: string, value: string | null) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          if (value) next.set(key, value)
          else next.delete(key)
          return next
        },
        { replace: true },
      )
    },
    [setParams],
  )

  const onSelect = useCallback((id: string | null) => setParam('selected', id), [setParam])

  const onLassoSelect = useCallback(
    (ids: string[]) => {
      // A lasso is a spatial question, and the honest answer is the set of
      // companies it enclosed. We surface them in the search explorer rather
      // than pretending the region is a category.
      if (ids.length === 0) return
      setParam('selected', ids[0])
      navigate({ pathname: '/search', search: `?${new URLSearchParams({ ids: ids.join('~') })}` })
    },
    [setParam, navigate],
  )

  const hovered = hoveredId ? dataset?.companiesById.get(hoveredId) : null
  const selected = selectedId ? dataset?.companiesById.get(selectedId) : null

  if (loading) {
    return (
      <div className="map">
        <div className="map__skeleton">
          <div className="map__spinner" aria-hidden="true" />
          <p className="muted">Loading the map…</p>
          <p className="faint">
            YC2Vec downloads a precomputed dataset once, then runs entirely in your browser.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="map">
      <FilterPanel />

      <section className="map__canvasWrap" aria-label="Company map">
        <div className="map__toolbar">
          <div className="map__colorModes" role="group" aria-label="Colour points by">
            <span className="faint map__toolbarLabel">Colour by</span>
            {COLOR_MODES.map((mode) => (
              <button
                key={mode.key}
                className={`pill pill--button${colorBy === mode.key ? ' pill--on' : ''}`}
                aria-pressed={colorBy === mode.key}
                onClick={() => setParam('color', mode.key)}
                disabled={mode.key === 'tag' && filters.tags.length === 0}
                title={
                  mode.key === 'tag' && filters.tags.length === 0
                    ? 'Select a semantic tag in the sidebar first'
                    : undefined
                }
              >
                {mode.label}
              </button>
            ))}
          </div>
          <div className="map__toolbarRight">
            <button className="btn" onClick={() => setFitToken((t) => t + 1)} disabled={visible.size === 0}>
              Fit to filter
            </button>
            {filters.tags.length > 0 && colorBy === 'tag' && (
              <button
                className="btn"
                onClick={() => setFilters((f) => ({ ...f, tags: f.tags.slice(0, -1) }))}
              >
                Drop last tag
              </button>
            )}
          </div>
        </div>

        {dataset && visible.size === 0 ? (
          <div className="map__empty">
            <h2>No companies match these filters</h2>
            <p className="muted">
              Try removing a semantic tag, widening the batch-year range, or switching tag matching
              from ALL to ANY.
            </p>
          </div>
        ) : (
          dataset && (
            <ScatterCanvas
              dataset={dataset}
              visible={visible}
              colorBy={colorBy}
              activeTagId={filters.tags[filters.tags.length - 1] ?? null}
              selectedId={selectedId}
              hoveredId={hoveredId}
              onHover={setHoveredId}
              onSelect={onSelect}
              onLassoSelect={onLassoSelect}
              fitToken={fitToken}
            />
          )
        )}

        {hovered && !selected && <HoverCard company={hovered} />}
      </section>

      {selected && (
        <CompanyPreview
          company={selected}
          onClose={() => setParam('selected', null)}
        />
      )}
    </div>
  )
}
