/**
 * Loading the published dataset.
 *
 * The site is static, so "the API" is a set of JSON files under a versioned
 * base path. Two constraints shape this module:
 *
 * 1. **Subpath safety.** GitHub Pages serves the project under `/<repo>/`, so
 *    every asset URL is resolved against Vite's `BASE_URL` rather than the
 *    document root. The same build works on a custom domain at `/`.
 * 2. **Progressive loading.** The map only needs `points.json` to paint, so it
 *    is fetched first and the heavier index resolves alongside it. Per-company
 *    detail is sharded and fetched on demand, then memoised.
 */

import type {
  ClustersPayload,
  Company,
  CompanyDetail,
  Dataset,
  PointsPayload,
  ReleaseManifest,
  SearchDoc,
  TagsPayload,
  TaxonomyPayload,
} from './types'

const BASE = import.meta.env.BASE_URL ?? '/'
const DATA_ROOT = `${BASE.replace(/\/$/, '')}/data/v1`

export class DataError extends Error {
  constructor(
    message: string,
    readonly url: string,
  ) {
    super(message)
    this.name = 'DataError'
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const url = `${DATA_ROOT}/${path}`
  let response: Response
  try {
    response = await fetch(url, { signal })
  } catch (cause) {
    throw new DataError(
      navigator.onLine === false
        ? 'You appear to be offline. YC2Vec needs to download its dataset once.'
        : `Could not reach ${path}.`,
      url,
    )
  }
  if (!response.ok) {
    throw new DataError(`${path} returned ${response.status}.`, url)
  }
  return (await response.json()) as T
}

/** Expand the short transport keys documented in the manifest's `key_map`. */
function expandCompany(row: Record<string, unknown>, pointIndex: number): Company {
  return {
    id: String(row.i),
    name: String(row.n),
    oneLiner: (row.o as string) ?? '',
    batch: (row.b as string) ?? null,
    batchYear: (row.y as number) ?? null,
    status: (row.s as string) ?? null,
    stage: (row.g as string) ?? null,
    industry: (row.d as string) ?? null,
    subindustry: (row.u as string) ?? null,
    regions: (row.r as string[]) ?? [],
    location: (row.l as string) ?? null,
    teamSize: (row.t as number) ?? null,
    website: (row.w as string) ?? null,
    ycUrl: (row.c as string) ?? null,
    logoUrl: (row.m as string) ?? null,
    sourceTags: (row.k as string[]) ?? [],
    topCompany: row.p === 1,
    isHiring: row.h === 1,
    nonprofit: row.z === 1,
    tagIds: (row.T as string[]) ?? [],
    tagScores: (row.S as number[]) ?? [],
    pointIndex,
  }
}

export async function loadDataset(signal?: AbortSignal): Promise<Dataset> {
  const [manifest, points] = await Promise.all([
    getJson<ReleaseManifest>('manifest.json', signal),
    getJson<PointsPayload>('points.json', signal),
  ])
  const [companiesRaw, tags, taxonomy, clusters] = await Promise.all([
    getJson<{ rows: Array<Record<string, unknown>> }>('companies.json', signal),
    getJson<TagsPayload>('tags.json', signal),
    getJson<TaxonomyPayload>('taxonomy.json', signal),
    getJson<ClustersPayload>('clusters.json', signal),
  ])

  const pointIndex = new Map<string, number>()
  points.ids.forEach((id, i) => pointIndex.set(id, i))

  const companies = companiesRaw.rows.map((row) =>
    expandCompany(row, pointIndex.get(String(row.i)) ?? -1),
  )
  const companiesById = new Map(companies.map((c) => [c.id, c]))
  const tagsById = new Map(tags.rows.map((t) => [t.tag_id, t]))

  return { manifest, points, companies, companiesById, tags, tagsById, taxonomy, clusters }
}

/**
 * Shard assignment. Must stay byte-identical to `shard_for` in
 * `pipeline/publish/browser.py`; the round-trip is covered by a unit test.
 */
export function shardFor(companyId: string, shards: number): number {
  let h = 0
  for (let i = 0; i < companyId.length; i += 1) {
    h = (Math.imul(h, 31) + companyId.charCodeAt(i)) >>> 0
  }
  return h % shards
}

const detailCache = new Map<number, Promise<Record<string, CompanyDetail>>>()

export async function loadCompanyDetail(
  companyId: string,
  shards: number,
  signal?: AbortSignal,
): Promise<CompanyDetail | null> {
  const shard = shardFor(companyId, shards)
  let pending = detailCache.get(shard)
  if (!pending) {
    pending = getJson<Record<string, CompanyDetail>>(`detail/${shard}.json`, signal)
    detailCache.set(shard, pending)
    // A failed shard must not poison the cache for the whole session.
    pending.catch(() => detailCache.delete(shard))
  }
  const shardData = await pending
  return shardData[companyId] ?? null
}

export async function loadSearchDocs(signal?: AbortSignal): Promise<SearchDoc[]> {
  const payload = await getJson<{ rows: SearchDoc[] }>('search/docs.json', signal)
  return payload.rows
}

export async function loadQuality(signal?: AbortSignal): Promise<Record<string, unknown> | null> {
  try {
    return await getJson<Record<string, unknown>>('quality.json', signal)
  } catch {
    return null
  }
}
