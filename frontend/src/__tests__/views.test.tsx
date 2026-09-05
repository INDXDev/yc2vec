import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { DatasetProvider } from '../lib/DatasetContext'
import { SearchExplorerView } from '../views/SearchExplorerView'
import { TagExplorerView } from '../views/TagExplorerView'
import { CompanyView } from '../views/CompanyView'
import { FilterPanel } from '../components/FilterPanel'

/**
 * These render the real views against a stubbed `fetch` serving the same
 * artifact shapes the pipeline publishes, so a change to the published format
 * that the UI cannot read fails here rather than in production.
 */

const manifest = {
  dataset_version: 'test', schema_version: '1', public_artifact_version: '1',
  pipeline_version: '1', ontology_version: '1', embedding_space_version: 'emb-1',
  projection_version: 'umap-1', generated_at: '2026-01-01T00:00:00Z', git_commit: 'abc',
  source_retrieved_at: null, source_last_updated: null, source_url: 'https://example.com',
  models: { chat: 'test-chat' }, prompt_hashes: {}, counts: {}, checksums: {},
  limitations: ['tags are inferred'], licenses: [], attribution: 'unofficial',
  detail_shards: 64, key_map: {},
}

const points = {
  version: 'umap-1', embedding_space_version: 'emb-1', count: 3,
  ids: ['c:1', 'c:2', 'c:3'], x: [0, 1, 2], y: [0, 1, 2], cluster: [0, 0, 1],
  year: [2020, 2015, 2024], note: 'UMAP is a lossy 2D projection.',
}

const companies = {
  key_map: {}, count: 3,
  rows: [
    { i: 'c:1', n: 'Stripe', o: 'Payments for the internet', b: 'Summer 2009', y: 2009,
      s: 'Active', d: 'B2B', T: ['payments'], S: [0.8] },
    { i: 'c:2', n: 'Airbnb', o: 'Book homes from local hosts', b: 'Winter 2009', y: 2009,
      s: 'Active', d: 'Consumer', T: ['marketplace'], S: [0.7] },
    { i: 'c:3', n: 'Untagged Co', o: 'Nothing assigned yet', b: 'Winter 2024', y: 2024,
      s: 'Active', d: 'B2B', T: [], S: [] },
  ],
}

const tags = {
  ontology_version: '1', count: 2, facets: ['business_model', 'workflow'],
  rows: [
    { tag_id: 'payments', name: 'Payments Infrastructure', facet: 'workflow',
      definition: 'The product moves or settles money between parties.', aliases: ['payment rails'],
      parents: [], prevalence: 1, support: 4, examples: ['Stripe'],
      cooccurring: [{ tag_id: 'marketplace', count: 1 }], by_year: { '2009': 1 } },
    { tag_id: 'marketplace', name: 'Marketplace Model', facet: 'business_model',
      definition: 'The company connects two sides of a transaction.', aliases: [],
      parents: [], prevalence: 1, support: 3, examples: ['Airbnb'],
      cooccurring: [], by_year: { '2009': 1 } },
  ],
}

const detail = {
  'c:1': {
    company_id: 'c:1', name: 'Stripe', one_liner: 'Payments for the internet',
    long_description: 'Stripe builds economic infrastructure for the internet.',
    website: 'https://stripe.com', yc_url: 'https://ycombinator.com/companies/stripe',
    logo_url: null, batch: 'Summer 2009', batch_year: 2009, status: 'Active', stage: 'Growth',
    team_size: 8000, industry: 'B2B', subindustry: 'B2B -> Finance', industries: ['B2B'],
    source_tags: ['Fintech'], regions: ['United States of America'], all_locations: 'San Francisco',
    former_names: [], top_company: true, is_hiring: false, nonprofit: false,
    metadata_document: 'Stripe is a Y Combinator company.', source_taxonomy_term_ids: [],
    tags: [{
      tag_id: 'payments', name: 'Payments Infrastructure', facet: 'workflow', value: 0.42,
      confidence: 0.86, raw_confidence: 0.9, weight: 0.49,
      rationale: 'The description states the company handles online payments.',
      evidence: [{ doc: 'c:1#yc_long_description', quote: 'economic infrastructure for the internet' }],
      shortlist_reason: 'retrieval',
    }],
    uncertain_tags: [{ tag_id: 'marketplace', name: 'Marketplace Model', notes: 'thin evidence' }],
    neighbors: {
      combined: [{ id: 'c:2', name: 'Airbnb', one_liner: 'Book homes', batch: 'Winter 2009',
                   score: 0.71, shared_metadata: ['status: Active'] }],
      metadata: [{ id: 'c:2', name: 'Airbnb', one_liner: 'Book homes', batch: 'Winter 2009',
                   score: 0.93, shared_metadata: ['batch: Winter 2009', 'status: Active'] }],
    },
    coordinates: { x: 0.1, y: -0.2, cluster: 0 },
  },
}

const searchDocs = {
  count: 3,
  rows: [
    { i: 'c:1', n: 'Stripe', a: '', o: 'Payments for the internet',
      e: 'Stripe builds economic infrastructure for the internet.',
      g: 'Payments Infrastructure', x: 'payment rails', d: 'B2B Fintech', b: 'Summer 2009 Active' },
    { i: 'c:2', n: 'Airbnb', a: '', o: 'Book homes from local hosts',
      e: 'A marketplace for lodging.', g: 'Marketplace Model', x: '', d: 'Consumer',
      b: 'Winter 2009 Active' },
    { i: 'c:3', n: 'Untagged Co', a: '', o: 'Nothing assigned yet', e: '', g: '', x: '',
      d: 'B2B', b: 'Winter 2024 Active' },
  ],
}

const ROUTES: Record<string, unknown> = {
  'manifest.json': manifest,
  'points.json': points,
  'companies.json': companies,
  'tags.json': tags,
  'taxonomy.json': { terms: [], mappings: [] },
  'clusters.json': { projection_version: 'umap-1', disclaimer: 'Algorithmic, not official.', rows: [] },
  'search/docs.json': searchDocs,
  'quality.json': { companies: 3, active_tags: 2 },
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    const key = Object.keys(ROUTES).find((k) => url.endsWith(k))
    if (key) return new Response(JSON.stringify(ROUTES[key]), { status: 200 })
    if (url.includes('/detail/')) return new Response(JSON.stringify(detail), { status: 200 })
    return new Response('not found', { status: 404 })
  }))
})

afterEach(() => vi.unstubAllGlobals())

function renderAt(path: string, element: React.ReactNode, routePath?: string) {
  // A Route path is a pathname; the query string belongs only to the entry.
  const pattern = routePath ?? path.split('?')[0]
  return render(
    <MemoryRouter initialEntries={[path]}>
      <DatasetProvider>
        <Routes>
          <Route path={pattern} element={element} />
        </Routes>
      </DatasetProvider>
    </MemoryRouter>,
  )
}

describe('search explorer', () => {
  it('lists every company and links onward', async () => {
    renderAt('/search', <SearchExplorerView />)
    expect(await screen.findByText('Stripe')).toBeInTheDocument()
    expect(screen.getByText('Airbnb')).toBeInTheDocument()
    const row = screen.getByText('Stripe').closest('tr') as HTMLElement
    expect(within(row).getByText('map')).toHaveAttribute('href', expect.stringContaining('selected=c%3A1'))
    expect(within(row).getByText('similar')).toHaveAttribute('href', expect.stringContaining('space=combined'))
  })

  it('narrows to keyword matches and shows why they matched', async () => {
    renderAt('/search?q=marketplace', <SearchExplorerView />)
    await waitFor(() => expect(screen.getByText('Airbnb')).toBeInTheDocument())
    await waitFor(() => expect(screen.queryByText('Stripe')).not.toBeInTheDocument())
    expect(await screen.findByText(/matched in/)).toBeInTheDocument()
  })

  it('announces the result count to assistive technology', async () => {
    renderAt('/search', <SearchExplorerView />)
    const status = await screen.findByRole('status')
    expect(status).toHaveTextContent(/3 companies/)
  })

  it('offers useful guidance when nothing matches', async () => {
    renderAt('/search?q=zzzzznotathing', <SearchExplorerView />)
    expect(await screen.findByText('No companies match')).toBeInTheDocument()
    expect(screen.getByText(/Switch to/)).toBeInTheDocument()
  })

  it('restricts to a map lasso selection', async () => {
    renderAt('/search?ids=c%3A2', <SearchExplorerView />)
    expect(await screen.findByText('Map selection')).toBeInTheDocument()
    expect(screen.getByText('Airbnb')).toBeInTheDocument()
    expect(screen.queryByText('Stripe')).not.toBeInTheDocument()
  })

  it('respects a tag filter from the URL', async () => {
    renderAt('/search?tag=payments', <SearchExplorerView />)
    await waitFor(() => expect(screen.getByText('Stripe')).toBeInTheDocument())
    expect(screen.queryByText('Airbnb')).not.toBeInTheDocument()
  })
})

describe('tag explorer', () => {
  it('lists tags with their facet and prevalence', async () => {
    renderAt('/tags', <TagExplorerView />)
    expect(await screen.findByText('Payments Infrastructure')).toBeInTheDocument()
    expect(screen.getByText('Marketplace Model')).toBeInTheDocument()
  })

  it('shows a tag definition, lineage and co-occurrence on deep link', async () => {
    renderAt('/tags/payments', <TagExplorerView />, '/tags/:tagId')
    expect(await screen.findByRole('heading', { name: 'Payments Infrastructure' })).toBeInTheDocument()
    expect(screen.getByText(/moves or settles money/)).toBeInTheDocument()
    expect(screen.getByText(/payment rails/)).toBeInTheDocument()
    expect(screen.getByText('Show these companies on the map')).toBeInTheDocument()
  })

  it('filters the tag list by text', async () => {
    const user = userEvent.setup()
    renderAt('/tags', <TagExplorerView />)
    await screen.findByText('Payments Infrastructure')
    await user.type(screen.getByLabelText(/Search tags/i), 'marketplace')
    await waitFor(() => expect(screen.queryByText('Payments Infrastructure')).not.toBeInTheDocument())
    expect(screen.getByText('Marketplace Model')).toBeInTheDocument()
  })
})

describe('company detail', () => {
  it('shows metadata, evidence and the rationale for each tag', async () => {
    renderAt('/company/c:1', <CompanyView />, '/company/:companyId')
    expect(await screen.findByRole('heading', { name: 'Stripe' })).toBeInTheDocument()
    expect(screen.getByText('Summer 2009')).toBeInTheDocument()
    expect(screen.getByText('Payments Infrastructure')).toBeInTheDocument()
    expect(screen.getByText(/description states the company handles online payments/)).toBeInTheDocument()
    // The phrase appears both in the description and as the quoted evidence
    // span; the evidence is the one rendered as a quotation.
    const quote = document.querySelector('.company__evidence q')
    expect(quote).toHaveTextContent('economic infrastructure for the internet')
  })

  it('shows undecided attributes rather than hiding them', async () => {
    renderAt('/company/c:1', <CompanyView />, '/company/:companyId')
    expect(await screen.findByText(/could not decide/)).toBeInTheDocument()
  })

  it('switches similarity mode and explains what it is showing', async () => {
    const user = userEvent.setup()
    renderAt('/company/c:1', <CompanyView />, '/company/:companyId')
    await screen.findByRole('heading', { name: 'Stripe' })

    expect(screen.getByText('0.710')).toBeInTheDocument()   // combined
    await user.click(screen.getByRole('tab', { name: 'Metadata' }))
    await waitFor(() => expect(screen.getByText('0.930')).toBeInTheDocument())
    expect(screen.getByText(/Batch, industry, region/)).toBeInTheDocument()
    // Metadata similarity is explained from shared fields, never invented.
    expect(screen.getByText(/batch: Winter 2009/)).toBeInTheDocument()
  })

  it('says so plainly when a similarity space has no neighbours', async () => {
    const user = userEvent.setup()
    renderAt('/company/c:1', <CompanyView />, '/company/:companyId')
    await screen.findByRole('heading', { name: 'Stripe' })
    await user.click(screen.getByRole('tab', { name: 'Sparse tag profile' }))
    expect(await screen.findByText(/No neighbours in this space/)).toBeInTheDocument()
  })

  it('surfaces the versions a technical reader needs', async () => {
    renderAt('/company/c:1', <CompanyView />, '/company/:companyId')
    await screen.findByRole('heading', { name: 'Stripe' })
    expect(screen.getByText('emb-1')).toBeInTheDocument()
    expect(screen.getByText('umap-1')).toBeInTheDocument()
  })

  it('reports a missing company instead of rendering an empty page', async () => {
    renderAt('/company/c:404', <CompanyView />, '/company/:companyId')
    expect(await screen.findByText('Company not found')).toBeInTheDocument()
  })
})

describe('filters', () => {
  it('exposes accessible controls and a live result count', async () => {
    renderAt('/', <FilterPanel />)
    expect(await screen.findByRole('complementary', { name: 'Filters' })).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('3 companies match')
    expect(screen.getByRole('group', { name: /Combine tags/ })).toBeInTheDocument()
    expect(screen.getByLabelText('Earliest batch year')).toBeInTheDocument()
  })

  it('toggles a tag filter and updates the count', async () => {
    const user = userEvent.setup()
    renderAt('/', <FilterPanel />)
    await screen.findByText('Payments Infrastructure')
    await user.click(screen.getByRole('checkbox', { name: /Payments Infrastructure/ }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('1 companies match'))
  })

  it('switches tag matching between ALL and ANY', async () => {
    const user = userEvent.setup()
    renderAt('/?tag=payments~marketplace', <FilterPanel />)
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('0 companies match'))
    await user.click(screen.getByRole('button', { name: 'ANY of' }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('2 companies match'))
  })

  it('clears every filter', async () => {
    const user = userEvent.setup()
    renderAt('/?tag=payments&top=1', <FilterPanel />)
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('0 companies match'))
    await user.click(screen.getByRole('button', { name: 'Clear all' }))
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('3 companies match'))
  })
})
