import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useDataset } from '../lib/DatasetContext'
import { toSearchParams } from '../lib/filters'
import type { Tag } from '../lib/types'
import '../styles/tags.css'

export function TagExplorerView() {
  const { tagId } = useParams<{ tagId: string }>()
  const { dataset, filters } = useDataset()
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [facet, setFacet] = useState<string | null>(null)

  const rows = dataset?.tags.rows ?? []

  const [assignedOnly, setAssignedOnly] = useState(false)

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return rows.filter((t) => {
      if (facet && t.facet !== facet) return false
      if (assignedOnly && t.prevalence === 0) return false
      if (!q) return true
      return (
        t.name.toLowerCase().includes(q) ||
        t.definition.toLowerCase().includes(q) ||
        t.aliases.some((a) => a.toLowerCase().includes(q))
      )
    })
  }, [rows, query, facet, assignedOnly])

  const unassigned = useMemo(() => rows.filter((t) => t.prevalence === 0).length, [rows])

  const selected = tagId ? dataset?.tagsById.get(tagId) : null

  const openInMap = (tag: Tag) => {
    const params = toSearchParams({ ...filters, tags: [tag.tag_id], tagMode: 'and' }, { color: 'tag' })
    navigate({ pathname: '/', search: `?${params.toString()}` })
  }

  if (!dataset) {
    return <div className="tags tags--message muted">Loading the ontology…</div>
  }

  if (rows.length === 0) {
    return (
      <div className="tags tags--message">
        <h1>No semantic tags in this release</h1>
        <p className="muted" style={{ maxWidth: '56ch' }}>
          The ontology is built incrementally by a local model. This release published no active
          tags, which happens when discovery has run but assignment has not yet grounded any tag in
          company evidence.
        </p>
        <Link className="btn" to="/">
          Back to the map
        </Link>
      </div>
    )
  }

  return (
    <div className="tags">
      <div className="tags__list scroll">
        <div className="tags__controls">
          <label className="sr-only" htmlFor="tag-explorer-search">
            Search tags
          </label>
          <input
            id="tag-explorer-search"
            type="search"
            placeholder={`Search ${rows.length} tags…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="tags__facets">
            <button
              className={`pill pill--button${facet === null ? ' pill--on' : ''}`}
              onClick={() => setFacet(null)}
            >
              all facets
            </button>
            {dataset.tags.facets.map((f) => (
              <button
                key={f}
                className={`pill pill--button${facet === f ? ' pill--on' : ''}`}
                onClick={() => setFacet(facet === f ? null : f)}
              >
                {f.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
          {unassigned > 0 && (
            <label className="tags__toggle">
              <input
                type="checkbox"
                checked={assignedOnly}
                onChange={() => setAssignedOnly((v) => !v)}
              />
              <span>Only tags with assignments</span>
            </label>
          )}
          <p className="faint mono tags__count" role="status" aria-live="polite">
            {filtered.length} tags
          </p>
        </div>

        <ul className="tags__items">
          {filtered.map((t) => (
            <li key={t.tag_id}>
              <Link
                to={`/tags/${encodeURIComponent(t.tag_id)}`}
                className={`tags__item${t.tag_id === tagId ? ' is-active' : ''}`}
              >
                <span className="tags__itemName">{t.name}</span>
                <span className="tags__itemFacet faint">{t.facet.replace(/_/g, ' ')}</span>
                <span
                  className={`mono faint${t.prevalence === 0 ? ' tags__zero' : ''}`}
                  title={t.prevalence === 0 ? 'No company has been judged against this tag yet' : undefined}
                >
                  {t.prevalence}
                </span>
              </Link>
            </li>
          ))}
          {filtered.length === 0 && <li className="faint tags__empty">No tag matches that text.</li>}
        </ul>
      </div>

      <div className="tags__detail scroll">
        {selected ? (
          <TagDetail tag={selected} onOpenInMap={openInMap} />
        ) : (
          <div className="tags__placeholder">
            <h1>Tag explorer</h1>
            <p className="muted">
              {rows.length.toLocaleString()} active semantic tags across{' '}
              {dataset.tags.facets.length} facets, discovered by a local model from the companies
              themselves rather than taken from a fixed list.
              {unassigned > 0 && (
                <>
                  {' '}
                  {(rows.length - unassigned).toLocaleString()} have been assigned to at least one
                  company so far; assignment is incremental and continues.
                </>
              )}
            </p>
            <p className="faint">
              Select a tag to see its definition, how prevalent it is, which tags it co-occurs with,
              and how it maps onto Y Combinator’s own categories.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function TagDetail({ tag, onOpenInMap }: { tag: Tag; onOpenInMap: (t: Tag) => void }) {
  const { dataset } = useDataset()
  const total = dataset?.companies.length ?? 0

  const years = Object.entries(tag.by_year).sort(([a], [b]) => a.localeCompare(b))
  const maxYear = Math.max(1, ...years.map(([, n]) => n))

  // The source-taxonomy relationship is the point of the mapping table: it
  // shows where the inferred ontology agrees with YC's categories and where it
  // cuts across them.
  const mappings = (dataset?.taxonomy.mappings ?? [])
    .filter((m) => m.tag_id === tag.tag_id)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 6)
  const termsById = new Map((dataset?.taxonomy.terms ?? []).map((t) => [t.term_id, t]))

  return (
    <article className="tagdetail">
      <header>
        <span className="pill pill--accent">{tag.facet.replace(/_/g, ' ')}</span>
        <h1 className="tagdetail__name">{tag.name}</h1>
        <p className="tagdetail__definition">{tag.definition}</p>
        {tag.aliases.length > 0 && (
          <p className="faint tagdetail__aliases">Also known as: {tag.aliases.join(', ')}</p>
        )}
      </header>

      <div className="tagdetail__stats">
        <Stat label="Companies" value={tag.prevalence.toLocaleString()} />
        <Stat label="Share of corpus" value={`${((tag.prevalence / Math.max(1, total)) * 100).toFixed(1)}%`} />
        <Stat label="Discovery support" value={tag.support.toLocaleString()} />
        <Stat label="Tag id" value={tag.tag_id} mono />
      </div>

      {tag.prevalence === 0 ? (
        <p className="tagdetail__pending panel">
          No company carries this tag yet. It was discovered and activated, but assignment is
          incremental and has not reached a company that exhibits it. This is a statement about
          coverage, not about the attribute.
        </p>
      ) : (
        <button className="btn btn--primary" onClick={() => onOpenInMap(tag)}>
          Show these companies on the map
        </button>
      )}

      {years.length > 1 && (
        <section className="tagdetail__section">
          <h2 className="tagdetail__sectionTitle">Distribution across batch years</h2>
          <ul className="tagdetail__years" aria-label="Companies with this tag by batch year">
            {years.map(([year, n]) => (
              <li key={year} title={`${year}: ${n} companies`}>
                <span className="tagdetail__bar" style={{ height: `${(n / maxYear) * 100}%` }} />
                <span className="tagdetail__yearLabel faint">{year.slice(2)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {tag.cooccurring.length > 0 && (
        <section className="tagdetail__section">
          <h2 className="tagdetail__sectionTitle">Co-occurring tags</h2>
          <ul className="tagdetail__chips">
            {tag.cooccurring.map((c) => {
              const other = dataset?.tagsById.get(c.tag_id)
              if (!other) return null
              return (
                <li key={c.tag_id}>
                  <Link className="pill pill--button" to={`/tags/${encodeURIComponent(c.tag_id)}`}>
                    {other.name} <span className="mono faint">{c.count}</span>
                  </Link>
                </li>
              )
            })}
          </ul>
        </section>
      )}

      {tag.examples.length > 0 && (
        <section className="tagdetail__section">
          <h2 className="tagdetail__sectionTitle">Companies that proposed this attribute</h2>
          <p className="faint tagdetail__note">
            These companies seeded the tag during discovery. Assignment is evaluated independently
            for every company, so this is lineage, not evidence.
          </p>
          <p>{tag.examples.join(', ')}</p>
        </section>
      )}

      <section className="tagdetail__section">
        <h2 className="tagdetail__sectionTitle">Relationship to YC’s own categories</h2>
        {mappings.length === 0 ? (
          <p className="muted tagdetail__note">
            No close relationship to a YC industry, tag or batch label was found — this attribute
            cuts across the source taxonomy rather than restating it.
          </p>
        ) : (
          <>
            <p className="faint tagdetail__note">
              YC2Vec keeps Y Combinator’s classifications untouched and records how its own tags
              relate to them. These relationships are computed from definition embeddings and are
              not human-reviewed.
            </p>
            <table className="data">
              <thead>
                <tr>
                  <th>YC term</th>
                  <th>kind</th>
                  <th>relation</th>
                  <th>similarity</th>
                </tr>
              </thead>
              <tbody>
                {mappings.map((m) => (
                  <tr key={m.term_id}>
                    <td>{termsById.get(m.term_id)?.name ?? m.term_id}</td>
                    <td className="faint">{termsById.get(m.term_id)?.kind}</td>
                    <td>{m.relation}</td>
                    <td className="mono faint">{m.similarity.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </article>
  )
}

function Stat({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="tagdetail__stat panel">
      <span className="tagdetail__statLabel">{label}</span>
      <strong className={mono ? 'mono' : ''}>{value}</strong>
    </div>
  )
}
