import { useMemo, useState } from 'react'
import { useDataset } from '../lib/DatasetContext'
import { activeFilterCount, facetCounts, isEmpty } from '../lib/filters'
import type { Company } from '../lib/types'

/**
 * The filter sidebar.
 *
 * Counts shown next to each option are computed against the *other* filters,
 * not the final result set, so a user can see how many companies an option
 * would add rather than only how many survive what they already chose.
 */
export function FilterPanel() {
  const { dataset, filters, setFilters, resetFilters, results } = useDataset()
  const [tagQuery, setTagQuery] = useState('')

  const options = useMemo(() => {
    const companies: Company[] = dataset?.companies ?? []
    return {
      batches: facetCounts(companies, (c) => (c.batch ? [c.batch] : [])),
      industries: facetCounts(companies, (c) => (c.industry ? [c.industry] : [])),
      statuses: facetCounts(companies, (c) => (c.status ? [c.status] : [])),
      stages: facetCounts(companies, (c) => (c.stage ? [c.stage] : [])),
      regions: facetCounts(companies, (c) => c.regions),
    }
  }, [dataset])

  const years = useMemo(() => {
    const present = (dataset?.companies ?? [])
      .map((c) => c.batchYear)
      .filter((y): y is number => y !== null)
    return present.length ? ([Math.min(...present), Math.max(...present)] as const) : ([2005, 2025] as const)
  }, [dataset])

  const visibleTags = useMemo(() => {
    const rows = dataset?.tags.rows ?? []
    const q = tagQuery.trim().toLowerCase()
    const matching = q
      ? rows.filter(
          (t) =>
            t.name.toLowerCase().includes(q) ||
            t.facet.includes(q) ||
            t.aliases.some((a) => a.toLowerCase().includes(q)),
        )
      : rows
    // Selected tags stay pinned at the top even when a query would hide them,
    // otherwise deselecting requires clearing the search first.
    const selected = rows.filter((t) => filters.tags.includes(t.tag_id))
    const rest = matching.filter((t) => !filters.tags.includes(t.tag_id))
    return [...selected, ...rest].slice(0, 60)
  }, [dataset, tagQuery, filters.tags])

  const toggle = (key: 'batches' | 'industries' | 'statuses' | 'stages' | 'regions' | 'tags' | 'facets', value: string) => {
    setFilters((f) => {
      const current = f[key] as string[]
      return {
        ...f,
        [key]: current.includes(value) ? current.filter((v) => v !== value) : [...current, value],
      }
    })
  }

  const count = activeFilterCount(filters)

  return (
    <aside className="filters scroll" aria-label="Filters">
      <div className="filters__head">
        <h2 className="filters__title">
          Filters {count > 0 && <span className="pill pill--accent">{count}</span>}
        </h2>
        <button className="btn" onClick={resetFilters} disabled={isEmpty(filters)}>
          Clear all
        </button>
      </div>
      <p className="filters__count muted" role="status" aria-live="polite">
        {results.length.toLocaleString()} companies match
      </p>

      <Section title="Semantic tags" defaultOpen>
        <div className="filters__tagmode" role="group" aria-label="Combine tags with">
          {(['and', 'or'] as const).map((mode) => (
            <button
              key={mode}
              className={`pill pill--button${filters.tagMode === mode ? ' pill--on' : ''}`}
              aria-pressed={filters.tagMode === mode}
              onClick={() => setFilters((f) => ({ ...f, tagMode: mode }))}
            >
              {mode === 'and' ? 'ALL of' : 'ANY of'}
            </button>
          ))}
        </div>
        <label className="sr-only" htmlFor="tag-filter-search">
          Find a semantic tag
        </label>
        <input
          id="tag-filter-search"
          type="search"
          className="filters__search"
          placeholder="Find a tag…"
          value={tagQuery}
          onChange={(e) => setTagQuery(e.target.value)}
        />
        <ul className="filters__list">
          {visibleTags.map((t) => (
            <li key={t.tag_id}>
              <label className="filters__option">
                <input
                  type="checkbox"
                  checked={filters.tags.includes(t.tag_id)}
                  onChange={() => toggle('tags', t.tag_id)}
                />
                <span className="filters__optionLabel" title={t.definition}>
                  {t.name}
                </span>
                <span className="faint mono">{t.prevalence}</span>
              </label>
            </li>
          ))}
          {visibleTags.length === 0 && (
            <li className="faint filters__empty">
              {dataset?.tags.rows.length
                ? 'No tag matches that text.'
                : 'No semantic tags in this release yet.'}
            </li>
          )}
        </ul>
      </Section>

      <Section title="Tag facet">
        <ul className="filters__list filters__list--chips">
          {(dataset?.tags.facets ?? []).map((facet) => (
            <li key={facet}>
              <button
                className={`pill pill--button${filters.facets.includes(facet) ? ' pill--on' : ''}`}
                aria-pressed={filters.facets.includes(facet)}
                onClick={() => toggle('facets', facet)}
              >
                {facet.replace(/_/g, ' ')}
              </button>
            </li>
          ))}
        </ul>
        <label className="filters__slider">
          <span>
            Minimum assignment confidence{' '}
            <span className="mono faint">{filters.minConfidence.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={filters.minConfidence}
            onChange={(e) =>
              setFilters((f) => ({ ...f, minConfidence: Number.parseFloat(e.target.value) }))
            }
          />
        </label>
      </Section>

      <Section title="Batch year">
        <div className="filters__range">
          <input
            type="range"
            min={years[0]}
            max={years[1]}
            value={filters.years?.[0] ?? years[0]}
            aria-label="Earliest batch year"
            onChange={(e) =>
              setFilters((f) => ({
                ...f,
                years: [Number.parseInt(e.target.value, 10), f.years?.[1] ?? years[1]],
              }))
            }
          />
          <input
            type="range"
            min={years[0]}
            max={years[1]}
            value={filters.years?.[1] ?? years[1]}
            aria-label="Latest batch year"
            onChange={(e) =>
              setFilters((f) => ({
                ...f,
                years: [f.years?.[0] ?? years[0], Number.parseInt(e.target.value, 10)],
              }))
            }
          />
          <span className="mono faint">
            {(filters.years?.[0] ?? years[0])}–{(filters.years?.[1] ?? years[1])}
          </span>
        </div>
      </Section>

      <CheckList title="Batch" values={options.batches} selected={filters.batches} onToggle={(v) => toggle('batches', v)} limit={24} />
      <CheckList title="YC industry" values={options.industries} selected={filters.industries} onToggle={(v) => toggle('industries', v)} limit={20} />
      <CheckList title="Status" values={options.statuses} selected={filters.statuses} onToggle={(v) => toggle('statuses', v)} />
      <CheckList title="Stage" values={options.stages} selected={filters.stages} onToggle={(v) => toggle('stages', v)} />
      <CheckList title="Region" values={options.regions} selected={filters.regions} onToggle={(v) => toggle('regions', v)} limit={20} />

      <Section title="Flags">
        <ul className="filters__list">
          {([
            ['topOnly', 'YC “top company” only'],
            ['hiringOnly', 'Currently hiring'],
            ['nonprofitOnly', 'Nonprofit'],
          ] as const).map(([key, label]) => (
            <li key={key}>
              <label className="filters__option">
                <input
                  type="checkbox"
                  checked={filters[key]}
                  onChange={() => setFilters((f) => ({ ...f, [key]: !f[key] }))}
                />
                <span className="filters__optionLabel">{label}</span>
              </label>
            </li>
          ))}
        </ul>
      </Section>
    </aside>
  )
}

function Section({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details className="filters__section" open={defaultOpen}>
      <summary className="filters__sectionTitle">{title}</summary>
      <div className="filters__sectionBody">{children}</div>
    </details>
  )
}

function CheckList({
  title,
  values,
  selected,
  onToggle,
  limit = 12,
}: {
  title: string
  values: Array<[string, number]>
  selected: string[]
  onToggle: (value: string) => void
  limit?: number
}) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? values : values.slice(0, limit)
  return (
    <Section title={title}>
      <ul className="filters__list">
        {shown.map(([value, count]) => (
          <li key={value}>
            <label className="filters__option">
              <input type="checkbox" checked={selected.includes(value)} onChange={() => onToggle(value)} />
              <span className="filters__optionLabel">{value}</span>
              <span className="faint mono">{count.toLocaleString()}</span>
            </label>
          </li>
        ))}
      </ul>
      {values.length > limit && (
        <button className="filters__more" onClick={() => setExpanded((v) => !v)}>
          {expanded ? 'Show fewer' : `Show all ${values.length}`}
        </button>
      )}
    </Section>
  )
}
