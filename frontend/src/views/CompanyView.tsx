import { useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { useDataset } from '../lib/DatasetContext'
import { loadCompanyDetail } from '../lib/data'
import { SIMILARITY_SPACES, type CompanyDetail, type SimilaritySpace } from '../lib/types'
import '../styles/company.css'

export function CompanyView() {
  const { companyId } = useParams<{ companyId: string }>()
  const { dataset } = useDataset()
  const [params, setParams] = useSearchParams()
  const [detail, setDetail] = useState<CompanyDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const space = (params.get('space') as SimilaritySpace) || 'combined'

  useEffect(() => {
    if (!companyId || !dataset) return
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    loadCompanyDetail(companyId, dataset.manifest.detail_shards, controller.signal)
      .then((d) => {
        setDetail(d)
        if (!d) setError('That company is not in this release.')
      })
      .catch((e: Error) => {
        if (e.name !== 'AbortError') setError(e.message)
      })
      .finally(() => setLoading(false))
    return () => controller.abort()
  }, [companyId, dataset])

  const grouped = useMemo(() => {
    const byFacet = new Map<string, CompanyDetail['tags']>()
    for (const t of detail?.tags ?? []) {
      const list = byFacet.get(t.facet) ?? []
      list.push(t)
      byFacet.set(t.facet, list)
    }
    return [...byFacet.entries()].sort((a, b) => b[1].length - a[1].length)
  }, [detail])

  if (loading) {
    return <div className="company company--message muted">Loading company…</div>
  }
  if (error || !detail) {
    return (
      <div className="company company--message">
        <h1>Company not found</h1>
        <p className="muted">{error}</p>
        <Link className="btn" to="/">
          Back to the map
        </Link>
      </div>
    )
  }

  const neighbors = detail.neighbors[space] ?? []
  const mapLink = `/?${new URLSearchParams({ selected: detail.company_id }).toString()}`

  return (
    <div className="company scroll">
      <article className="company__inner">
        <header className="company__header">
          <div className="company__identity">
            {detail.logo_url && (
              /*
               * Logos are hotlinked from the source record rather than
               * redistributed, and are decorative: the name is always present
               * as text, so a blocked image costs nothing.
               */
              <img
                className="company__logo"
                src={detail.logo_url}
                alt=""
                loading="lazy"
                referrerPolicy="no-referrer"
                onError={(e) => {
                  e.currentTarget.style.display = 'none'
                }}
              />
            )}
            <div>
              <h1 className="company__name">{detail.name}</h1>
              {detail.one_liner && <p className="company__line">{detail.one_liner}</p>}
              {detail.former_names.length > 0 && (
                <p className="faint company__former">
                  Formerly {detail.former_names.join(', ')}
                </p>
              )}
            </div>
          </div>
          <div className="company__links">
            <Link className="btn" to={mapLink}>
              Show on map
            </Link>
            {detail.website && (
              <a className="btn" href={detail.website} target="_blank" rel="noreferrer noopener">
                Website ↗
              </a>
            )}
            <a className="btn" href={detail.yc_url} target="_blank" rel="noreferrer noopener">
              YC page ↗
            </a>
          </div>
        </header>

        <section className="company__meta panel">
          <dl className="company__facts">
            <Fact label="Batch" value={detail.batch} />
            <Fact label="Status" value={detail.status} />
            <Fact label="Stage" value={detail.stage} />
            <Fact label="Team size" value={detail.team_size?.toLocaleString() ?? null} />
            <Fact label="YC industry" value={detail.industry} />
            <Fact label="Sub-industry" value={detail.subindustry} />
            <Fact label="Location" value={detail.all_locations} />
            <Fact label="Regions" value={detail.regions.join(', ') || null} />
          </dl>
          {detail.source_tags.length > 0 && (
            <div className="company__sourceTags">
              <span className="company__factLabel">YC source tags</span>
              <div className="company__pills">
                {detail.source_tags.map((t) => (
                  <span key={t} className="pill">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>

        {detail.long_description && (
          <section className="company__section">
            <h2 className="company__sectionTitle">Description</h2>
            <p className="company__body">{detail.long_description}</p>
            <p className="faint company__provenance">
              Source text from the public YC company record, via yc-oss/api.
            </p>
          </section>
        )}

        <section className="company__section">
          <h2 className="company__sectionTitle">
            Semantic tags{' '}
            <span className="faint">
              {detail.tags.length} assigned{detail.tags.length ? ', grouped by facet' : ''}
            </span>
          </h2>
          {grouped.length === 0 ? (
            <p className="muted company__body">
              No semantic tags were assigned to this company in the current release. Tag assignment
              is incremental: companies are enriched over successive pipeline runs, and a company
              with weak public text may legitimately receive none.
            </p>
          ) : (
            grouped.map(([facet, tags]) => (
              <div key={facet} className="company__facet">
                <h3 className="company__facetTitle">{facet.replace(/_/g, ' ')}</h3>
                <ul className="company__tagList">
                  {tags.map((t) => (
                    <li key={t.tag_id} className="company__tag panel">
                      <div className="company__tagHead">
                        <Link className="company__tagName" to={`/tags/${encodeURIComponent(t.tag_id)}`}>
                          {t.name}
                        </Link>
                        <ConfidenceBar value={t.confidence} weight={t.weight} />
                      </div>
                      {t.rationale && <p className="company__rationale">{t.rationale}</p>}
                      {t.evidence.length > 0 && (
                        <ul className="company__evidence">
                          {t.evidence.map((e, i) => (
                            <li key={i}>
                              <span className="mono faint">{e.doc.split('#')[1] ?? e.doc}</span>
                              <q>{e.quote}</q>
                            </li>
                          ))}
                        </ul>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))
          )}
          {detail.uncertain_tags.length > 0 && (
            <details className="company__uncertain">
              <summary>
                {detail.uncertain_tags.length} attributes the model could not decide
              </summary>
              <p className="faint">
                These were shortlisted but the evidence was insufficient. They are shown so the
                dataset does not look more certain than it is.
              </p>
              <div className="company__pills">
                {detail.uncertain_tags.map((t) => (
                  <span key={t.tag_id} className="pill" title={t.notes ?? undefined}>
                    {t.name}
                  </span>
                ))}
              </div>
            </details>
          )}
        </section>

        <section className="company__section">
          <h2 className="company__sectionTitle">Most similar companies</h2>
          <p className="faint company__note">
            Ranked by cosine similarity in the full high-dimensional space, not by distance on the
            2D map.
          </p>
          <div className="company__spaces" role="tablist" aria-label="Similarity mode">
            {SIMILARITY_SPACES.map((s) => (
              <button
                key={s.key}
                role="tab"
                aria-selected={space === s.key}
                className={`pill pill--button${space === s.key ? ' pill--on' : ''}`}
                title={s.blurb}
                onClick={() =>
                  setParams(
                    (prev) => {
                      const next = new URLSearchParams(prev)
                      next.set('space', s.key)
                      return next
                    },
                    { replace: true },
                  )
                }
              >
                {s.label}
              </button>
            ))}
          </div>
          <p className="muted company__spaceBlurb">
            {SIMILARITY_SPACES.find((s) => s.key === space)?.blurb}
          </p>

          {neighbors.length === 0 ? (
            <p className="muted company__body">
              No neighbours in this space. The {space === 'tags' || space === 'sparse_tags' ? 'tag' : ''}{' '}
              representation requires assigned tags, which this company does not have yet.
            </p>
          ) : (
            <ol className="company__neighbors">
              {neighbors.map((n) => {
                // Resolved from the company index the app already holds.
                const other = dataset?.companiesById.get(n.id)
                return (
                <li key={n.id} className="company__neighbor panel">
                  <div className="company__neighborHead">
                    <Link to={`/company/${encodeURIComponent(n.id)}?space=${space}`}>
                      {other?.name ?? n.id}
                    </Link>
                    <span className="mono faint">{n.score.toFixed(3)}</span>
                  </div>
                  {/* 84 company names are shared by more than one YC company,
                      so the batch is what makes a neighbour identifiable. */}
                  {other?.batch && <p className="company__neighborBatch faint">{other.batch}</p>}
                  {other?.oneLiner && <p className="company__neighborLine">{other.oneLiner}</p>}
                  {n.shared_tags && n.shared_tags.length > 0 && (
                    <p className="company__why">
                      <span className="faint">Shared tags:</span> {n.shared_tags.join(', ')}
                    </p>
                  )}
                  {n.shared_metadata && n.shared_metadata.length > 0 && (
                    <p className="company__why">
                      <span className="faint">Shared metadata:</span> {n.shared_metadata.join(' · ')}
                    </p>
                  )}
                </li>
                )
              })}
            </ol>
          )}
        </section>

        <section className="company__section company__technical">
          <h2 className="company__sectionTitle">Technical details</h2>
          <dl className="company__facts">
            <Fact label="Company id" value={detail.company_id} mono />
            <Fact
              label="Map coordinates"
              value={
                detail.coordinates
                  ? `${detail.coordinates.x.toFixed(3)}, ${detail.coordinates.y.toFixed(3)} (cluster ${detail.coordinates.cluster})`
                  : null
              }
              mono
            />
            <Fact label="Embedding space" value={dataset?.manifest.embedding_space_version ?? null} mono />
            <Fact label="Projection" value={dataset?.manifest.projection_version ?? null} mono />
            <Fact label="Dataset version" value={dataset?.manifest.dataset_version ?? null} mono />
            <Fact
              label="Data generated"
              value={
                dataset?.manifest.generated_at
                  ? new Date(dataset.manifest.generated_at).toISOString().slice(0, 16).replace('T', ' ') + ' UTC'
                  : null
              }
              mono
            />
          </dl>
          <details className="company__metadataDoc">
            <summary>Metadata document used for metadata similarity</summary>
            <p className="company__body">{detail.metadata_document}</p>
          </details>
        </section>
      </article>
    </div>
  )
}

function Fact({ label, value, mono = false }: { label: string; value: string | null; mono?: boolean }) {
  if (!value) return null
  return (
    <div className="company__fact">
      <dt className="company__factLabel">{label}</dt>
      <dd className={`company__factValue${mono ? ' mono' : ''}`}>{value}</dd>
    </div>
  )
}

function ConfidenceBar({ value, weight }: { value: number; weight: number }) {
  return (
    <span
      className="confbar"
      title={`Calibrated confidence ${value.toFixed(2)} × information weight ${weight.toFixed(2)}`}
    >
      <span className="confbar__track">
        <span className="confbar__fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
      <span className="mono faint">{value.toFixed(2)}</span>
    </span>
  )
}
