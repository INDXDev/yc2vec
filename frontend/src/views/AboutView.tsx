import { useEffect, useState } from 'react'
import { useDataset } from '../lib/DatasetContext'
import { loadQuality } from '../lib/data'
import '../styles/about.css'

export function AboutView() {
  const { dataset } = useDataset()
  const [quality, setQuality] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    loadQuality(controller.signal).then(setQuality)
    return () => controller.abort()
  }, [])

  const m = dataset?.manifest
  const num = (k: string): string => {
    const v = quality?.[k]
    return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: 4 }) : '—'
  }

  return (
    <div className="about scroll">
      <article className="about__inner">
        <div className="about__notice panel">
          <strong>Unofficial project.</strong> YC2Vec is an independent open-data experiment. It is
          not affiliated with, endorsed by, or connected to Y Combinator. Company records come from
          the open-source{' '}
          <a href="https://github.com/yc-oss/api" target="_blank" rel="noreferrer noopener">
            yc-oss/api
          </a>{' '}
          project, which republishes YC’s public company index. Semantic tags are inferred by a
          language model and are not verified by YC or by the companies described.
        </div>

        <h1>How YC2Vec is built</h1>
        <p className="about__lede">
          YC2Vec asks a simple question: if you described every Y Combinator company by what it
          actually does — who buys it, what workflow it replaces, how it is delivered — rather than
          by which industry bucket it landed in, what shape would the ecosystem have?
        </p>

        <Section title="1. Collection">
          <p>
            Structured records come from <code>yc-oss/api</code>, a static JSON mirror of YC’s
            public company index. We pin the source URL, record the retrieval timestamp and the
            upstream <code>last_updated</code> stamp, and keep the raw payload verbatim so every
            derived field can be traced back to it.
          </p>
          <p>
            Optional website enrichment is <strong>off by default</strong>. When it is switched on
            it obeys <code>robots.txt</code>, identifies itself with a contact URL, throttles
            per-domain, caps response size, refuses private and cloud-metadata addresses,
            re-validates every redirect, and never executes page JavaScript. Only extracted main
            text is kept, and only in a local cache — raw page bodies are never redistributed.
          </p>
        </Section>

        <Section title="2. Ontology discovery">
          <p>
            A local model is shown small, deliberately <em>diverse</em> batches of companies and
            asked what reusable semantic attributes distinguish them. Batching across unrelated
            industries is what makes the result cut across obvious verticals instead of restating
            them.
          </p>
          <p>
            Proposals are candidates, not tags. They are normalised, matched against existing
            aliases, and compared by definition embedding. Highly similar pairs merge
            deterministically; borderline pairs are adjudicated by the model; anything still
            ambiguous enters a review queue and is <strong>never merged silently</strong>. A
            candidate only becomes active once it has enough independent support and a usable
            definition.
          </p>
          <p>
            Tag ids are stable slugs minted once and frozen. Renaming a tag never changes its id, and
            a merge is recorded as a migration rather than a deletion, so historical assignments
            stay interpretable.
          </p>
        </Section>

        <Section title="3. Assignment">
          <p>
            Evaluating every company against every tag would be mostly wasted work, so each company
            gets a shortlist from retrieval similarity, facet priors, literal alias hits, metadata
            rules and tag hierarchy — plus a set of deliberately plausible{' '}
            <strong>hard negatives</strong>. The share of hard negatives the model accepts is our
            honest estimate of its false-positive rate; it is published below rather than hidden.
          </p>
          <p>
            Each pair is judged independently against the tag’s own definition and the supplied
            evidence. The model must answer <code>yes</code>, <code>no</code> or{' '}
            <code>uncertain</code> — weak evidence is never rounded into a decision — and must quote
            verbatim spans to support a positive. Quotes that do not literally occur in the evidence
            are discarded, and a positive left without verifiable evidence is downgraded to
            uncertain.
          </p>
          <p>
            Each quote is labelled with where it came from. A tag justified from the metadata
            document was inferred from structured fields — industry, batch, region — rather than
            from anything the company wrote about itself, and the company page says so rather than
            presenting the two as equivalent.
          </p>
          <p>
            The published feature value is{' '}
            <code>calibrated_confidence × information_weight</code>, where the information weight
            damps tags that apply to almost everything. Binary presence and raw confidence are
            stored separately, and missing, <code>no</code> and <code>uncertain</code> are never
            conflated.
          </p>
        </Section>

        <Section title="4. Vectors and the map">
          <p>
            Four representations are built per company with the same local embedding model:
            description text, a deterministic serialisation of structured metadata, the definitions
            of its assigned tags, and a combined vector. Because every component lives in one space,
            the combination is a weighted sum of normalised vectors, renormalised — not a
            concatenation of incompatible spaces. Each component is preserved, which is why the
            company page can switch between similarity modes and explain which one it is using.
          </p>
          <p>
            Nearest neighbours are precomputed exactly in every space and shipped as static files.
            The 2D map is UMAP, and it is <strong>only</strong> for visualisation: chart distance is
            a lossy compression of a {String(quality?.dim ?? '')}high-dimensional geometry, so
            similarity questions are always answered from the precomputed lists. Clusters are
            algorithmic groupings labelled by over-represented tags, not official categories.
          </p>
        </Section>

        <Section title="5. What is inferred versus what is sourced">
          <table className="data about__table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Origin</th>
                <th>Trust</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Company metadata, YC industries, batches, tags</td>
                <td>Public source records via yc-oss/api</td>
                <td>Reproduced exactly; never renamed or merged by us</td>
              </tr>
              <tr>
                <td>Semantic tag ontology and assignments</td>
                <td>Local language model</td>
                <td>Inferred; every assignment carries evidence and a rationale</td>
              </tr>
              <tr>
                <td>Embeddings, neighbours, UMAP, clusters</td>
                <td>Deterministic computation over the above</td>
                <td>Reproducible from the recorded seeds and versions</td>
              </tr>
              <tr>
                <td>Source-taxonomy ↔ tag mappings</td>
                <td>Embedding similarity</td>
                <td>Comparison aid; not human-reviewed</td>
              </tr>
            </tbody>
          </table>
        </Section>

        <Section title="6. Quality">
          <p className="muted">
            These numbers describe the current release. None of them gate publication on their own;
            they are here so you can decide how much to trust the dataset.
          </p>
          <div className="about__stats">
            <Stat label="Companies" value={num('companies')} />
            <Stat label="Active tags" value={num('active_tags')} />
            <Stat label="Assignments" value={num('assignments')} />
            <Stat label="Tags per company" value={num('assignments_per_company')} />
            <Stat label="Evidence coverage" value={num('evidence_coverage')} />
            <Stat label="Uncertain rate" value={num('uncertain_rate')} />
            <Stat label="Hard-negative acceptance" value={num('hard_negative_acceptance_rate')} />
            <Stat label="Orphan-tag rate" value={num('orphan_tag_rate')} />
            <Stat label="Duplicate-tag rate" value={num('duplicate_tag_rate')} />
          </div>
        </Section>

        <Section title="7. Limitations">
          <ul className="about__list">
            {(m?.limitations ?? []).map((l) => (
              <li key={l}>{l}</li>
            ))}
            <li>
              Tag assignment is incremental. A company with no semantic tags has not yet been
              processed, or its public text was too thin to ground any attribute — it is not a
              statement about the company.
            </li>
            <li>
              The model runs locally at low temperature with a fixed seed, but generation is not
              bit-for-bit deterministic across hardware and runtime versions.
            </li>
          </ul>
        </Section>

        <Section title="8. “Zero token cost”">
          <p>
            Every model call in this project runs on local hardware through Ollama. There are no
            metered API-token fees. That is <em>not</em> the same as free: it costs hardware,
            electricity, bandwidth and maintenance, and the full quantised model weights must be
            resident in memory for the whole run.
          </p>
        </Section>

        <Section title="9. Privacy and attribution">
          <p>
            YC2Vec describes companies, not individuals. It stores no founder names beyond what
            appears in a company’s own public description, sets no cookies, and sends no analytics —
            the site is static files and your browser.
          </p>
          <p>{m?.attribution}</p>
        </Section>

        {m && (
          <Section title="Release manifest">
            <dl className="about__manifest">
              <ManifestRow label="Dataset version" value={m.dataset_version} />
              <ManifestRow label="Schema / artifact version" value={`${m.schema_version} / ${m.public_artifact_version}`} />
              <ManifestRow label="Pipeline version" value={m.pipeline_version} />
              <ManifestRow label="Ontology version" value={m.ontology_version} />
              <ManifestRow label="Embedding space" value={m.embedding_space_version} />
              <ManifestRow label="Projection" value={m.projection_version} />
              <ManifestRow label="Chat model" value={String(m.models?.chat ?? '—')} />
              <ManifestRow label="Embedding model" value={String(m.models?.embedding ?? '—')} />
              <ManifestRow label="Prompt hashes" value={Object.entries(m.prompt_hashes ?? {}).map(([k, v]) => `${k}=${v}`).join(' ')} />
              <ManifestRow label="Git commit" value={m.git_commit ?? '—'} />
              <ManifestRow label="Source last updated" value={m.source_last_updated ?? '—'} />
              <ManifestRow label="Generated at" value={m.generated_at} />
            </dl>
            <p className="faint">
              Licences:{' '}
              {(m.licenses ?? [])
                .map((l) => `${l.name}${l.spdx ? ` (${l.spdx})` : ''}`)
                .join(' · ')}
            </p>
          </Section>
        )}
      </article>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="about__section">
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="about__stat panel">
      <span className="about__statLabel">{label}</span>
      <strong className="mono">{value}</strong>
    </div>
  )
}

function ManifestRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className="mono">{value}</dd>
    </>
  )
}
