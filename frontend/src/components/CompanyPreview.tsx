import { Link, useSearchParams } from 'react-router-dom'
import { useDataset } from '../lib/DatasetContext'
import type { Company } from '../lib/types'

/**
 * The side panel shown when a point on the map is selected.
 *
 * Deliberately a summary, not the full record: it answers "what is this dot?"
 * without a navigation, and links to the full detail view for evidence and
 * neighbours.
 */
export function CompanyPreview({ company, onClose }: { company: Company; onClose: () => void }) {
  const { dataset } = useDataset()
  const [params] = useSearchParams()

  const tags = company.tagIds
    .map((id, i) => ({ tag: dataset?.tagsById.get(id), value: company.tagScores[i] ?? 0 }))
    .filter((t): t is { tag: NonNullable<typeof t.tag>; value: number } => Boolean(t.tag))
    .slice(0, 10)

  return (
    <aside className="preview scroll" aria-label={`${company.name} summary`}>
      <div className="preview__head">
        <h2 className="preview__name">{company.name}</h2>
        <button className="preview__close" onClick={onClose} aria-label="Close preview">
          ×
        </button>
      </div>

      {company.oneLiner && <p className="preview__line">{company.oneLiner}</p>}

      <div className="preview__meta">
        {company.batch && <span className="pill">{company.batch}</span>}
        {company.status && <span className="pill">{company.status}</span>}
        {company.stage && <span className="pill">{company.stage}</span>}
        {company.industry && <span className="pill">{company.industry}</span>}
        {company.teamSize !== null && <span className="pill">{company.teamSize} people</span>}
        {company.topCompany && <span className="pill pill--accent">YC top company</span>}
      </div>

      {company.location && <p className="faint preview__location">{company.location}</p>}

      {tags.length > 0 ? (
        <section className="preview__section">
          <h3 className="preview__sectionTitle">Semantic tags</h3>
          <ul className="preview__tags">
            {tags.map(({ tag, value }) => (
              <li key={tag.tag_id}>
                <Link className="pill pill--button" to={`/tags/${encodeURIComponent(tag.tag_id)}`}>
                  {tag.name}
                  <span className="mono faint">{value.toFixed(2)}</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <p className="faint preview__none">
          No semantic tags assigned in this release. Tag assignment is incremental, so companies
          are enriched over successive runs.
        </p>
      )}

      <div className="preview__actions">
        <Link
          className="btn btn--primary"
          to={`/company/${encodeURIComponent(company.id)}?${params.toString()}`}
        >
          Full profile & similar companies
        </Link>
        {company.ycUrl && (
          <a className="btn" href={company.ycUrl} target="_blank" rel="noreferrer noopener">
            YC page ↗
          </a>
        )}
      </div>
    </aside>
  )
}
