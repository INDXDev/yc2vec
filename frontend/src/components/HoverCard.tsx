import type { Company } from '../lib/types'

/** Lightweight preview shown while the pointer rests on a point. */
export function HoverCard({ company }: { company: Company }) {
  return (
    <div className="hovercard panel">
      <strong className="hovercard__name">{company.name}</strong>
      {company.oneLiner && <p className="hovercard__line">{company.oneLiner}</p>}
      <div className="hovercard__meta">
        {company.batch && <span className="pill">{company.batch}</span>}
        {company.industry && <span className="pill">{company.industry}</span>}
        {company.status && <span className="pill">{company.status}</span>}
      </div>
    </div>
  )
}
