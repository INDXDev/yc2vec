import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import createScatterplot from 'regl-scatterplot'
import type { Dataset } from '../lib/types'

/**
 * The DNA map canvas.
 *
 * WebGL, one draw call for the whole corpus. Rendering a DOM node per company
 * would be thousands of nodes and would not survive panning; regl-scatterplot
 * keeps the points in buffers and only uploads when the colouring or the
 * filtered set actually changes.
 *
 * Filtering is expressed as *opacity*, not as removal: keeping the excluded
 * points faintly visible preserves the shape of the whole corpus, so a filter
 * reads as "here is where these sit" rather than "here are some dots".
 */

export type ColorBy = 'year' | 'industry' | 'cluster' | 'tag'

/**
 * Categorical ramp. Ordered so adjacent entries stay distinguishable, and
 * chosen for contrast against the dark canvas rather than for prettiness.
 */
const CATEGORY_COLORS = [
  '#5ee0c0', '#7aa2f7', '#f0b04a', '#f2686b', '#bb9af7', '#7dcfff',
  '#9ece6a', '#ff9e64', '#e0af68', '#41a6b5', '#c0caf5', '#ff75a0',
  '#73daca', '#b4f9f8', '#d4a0ff', '#f7768e', '#a3be8c', '#88c0d0',
  '#ebcb8b', '#d08770', '#8fbcbb', '#5e81ac', '#b48ead', '#a3d9a5',
]

/** Sequential ramp for batch year: older is cool, newer is warm. */
const YEAR_COLORS = [
  '#2c3d63', '#35538a', '#3d6ba8', '#4a86bd', '#5ea3c9', '#7cc0cb',
  '#a4d8c0', '#cfe8ab', '#f0e08a', '#f5c168', '#ef9c52', '#e37447',
  '#d24f43', '#b83440',
]

const DIMMED = '#2a3040'

export interface ScatterProps {
  dataset: Dataset
  /** Indices into the points arrays that pass the current filters. */
  visible: Set<number>
  colorBy: ColorBy
  activeTagId: string | null
  selectedId: string | null
  hoveredId: string | null
  onHover: (companyId: string | null) => void
  onSelect: (companyId: string | null) => void
  onLassoSelect: (companyIds: string[]) => void
  fitToken: number
}

interface Legend {
  label: string
  entries: Array<{ label: string; color: string; count: number }>
}

export function ScatterCanvas(props: ScatterProps) {
  const {
    dataset, visible, colorBy, activeTagId, selectedId,
    onHover, onSelect, onLassoSelect, fitToken,
  } = props

  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const plotRef = useRef<ReturnType<typeof createScatterplot> | null>(null)
  const [ready, setReady] = useState(false)

  const ids = dataset.points.ids

  const sizeRange = useMemo<[number, number]>(() => {
    const n = ids.length
    if (n <= 200) return [3, 9]
    if (n <= 1500) return [2.4, 6]
    return [1.8, 4.2]
  }, [ids.length])

  // Colour assignment. Each point gets an index into a palette; the palette
  // itself is uploaded once per colouring change rather than per point.
  const { colorIndex, palette, legend } = useMemo(
    () => computeColors(dataset, colorBy, activeTagId),
    [dataset, colorBy, activeTagId],
  )

  // Columnar input. regl-scatterplot also accepts an array of [x, y, ...] rows,
  // but that would allocate one short array per company on every filter change;
  // the column form uploads typed arrays straight into the buffers.
  //
  // valueA selects the palette entry and valueB drives both size and opacity,
  // so a filtered-out point fades and shrinks rather than disappearing — which
  // keeps the shape of the whole corpus visible behind any filter.
  const points = useMemo(() => {
    const n = ids.length
    const valueA = new Float32Array(n)
    const valueB = new Float32Array(n)
    const dimmed = (palette.length - 1) / Math.max(1, palette.length - 1)
    for (let i = 0; i < n; i += 1) {
      const shown = visible.has(i)
      // valueA is normalised to [0, 1]: regl-scatterplot maps it across the
      // palette rather than treating it as a raw index.
      valueA[i] = shown
        ? colorIndex[i] / Math.max(1, palette.length - 1)
        : dimmed
      valueB[i] = shown ? 1 : 0
    }
    return {
      x: Float32Array.from(dataset.points.x),
      y: Float32Array.from(dataset.points.y),
      valueA,
      valueB,
    }
  }, [dataset, ids.length, visible, colorIndex, palette.length])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const plot = createScatterplot({
      canvas,
      width: container.clientWidth,
      height: container.clientHeight,
      pointSize: 3.2,
      opacity: 0.85,
      lassoMinDelay: 12,
      lassoInitiator: true,
      showReticle: true,
      reticleColor: [1, 1, 1, 0.16],
      backgroundColor: getComputedStyle(document.body).backgroundColor || '#0a0c10',
    })
    plotRef.current = plot
    setReady(true)

    const resize = () => {
      plot.set({ width: container.clientWidth, height: container.clientHeight })
    }
    const observer = new ResizeObserver(resize)
    observer.observe(container)

    return () => {
      observer.disconnect()
      plot.destroy()
      plotRef.current = null
    }
  }, [])

  // Event handlers are re-subscribed when the callbacks change, so they never
  // close over a stale filtered set.
  useEffect(() => {
    const plot = plotRef.current
    if (!plot) return
    const hover = (i: number) => onHover(ids[i] ?? null)
    const unhover = () => onHover(null)
    const select = ({ points: sel }: { points: number[] }) => {
      if (sel.length === 1) onSelect(ids[sel[0]] ?? null)
      else if (sel.length > 1) onLassoSelect(sel.map((i) => ids[i]).filter(Boolean))
    }
    const deselect = () => onSelect(null)
    plot.subscribe('pointOver', hover)
    plot.subscribe('pointOut', unhover)
    plot.subscribe('select', select)
    plot.subscribe('deselect', deselect)
    return () => {
      plot.unsubscribe('pointOver', hover)
      plot.unsubscribe('pointOut', unhover)
      plot.unsubscribe('select', select)
      plot.unsubscribe('deselect', deselect)
    }
  }, [ids, onHover, onSelect, onLassoSelect])

  useEffect(() => {
    const plot = plotRef.current
    if (!plot || !ready) return
    let cancelled = false
    plot.set({
      colorBy: 'valueA',
      sizeBy: 'valueB',
      pointColor: palette,
      // Point size is scaled to the corpus: a 60-company fixture needs larger
      // marks to read as a map, while the full corpus needs small ones to stay
      // legible where it is dense.
      pointSize: sizeRange,
      opacityBy: 'valueB',
      opacity: [0.16, 0.92],
    })
    // Draws must not overlap: a rapid filter change would otherwise queue a
    // second upload before the first finished.
    void (async () => {
      await plot.draw(points as unknown as Parameters<typeof plot.draw>[0])
      if (cancelled) return
    })()
    return () => {
      cancelled = true
    }
  }, [points, palette, ready, sizeRange])

  useEffect(() => {
    const plot = plotRef.current
    if (!plot || !ready || !selectedId) return
    const idx = ids.indexOf(selectedId)
    if (idx >= 0) plot.select([idx], { preventEvent: true })
  }, [selectedId, ids, ready])

  useEffect(() => {
    const plot = plotRef.current
    if (!plot || !ready || fitToken === 0) return
    const idx = [...visible]
    if (idx.length === 0) return
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const i of idx) {
      minX = Math.min(minX, dataset.points.x[i])
      maxX = Math.max(maxX, dataset.points.x[i])
      minY = Math.min(minY, dataset.points.y[i])
      maxY = Math.max(maxY, dataset.points.y[i])
    }
    const pad = Math.max(0.12, (maxX - minX) * 0.08)
    plot.zoomToArea(
      { x: minX - pad, y: minY - pad, width: maxX - minX + pad * 2, height: maxY - minY + pad * 2 },
      { transition: true },
    )
  }, [fitToken, visible, dataset, ready])

  const reset = useCallback(() => plotRef.current?.reset(), [])

  return (
    <div className="scatter" ref={containerRef}>
      <canvas ref={canvasRef} className="scatter__canvas" />
      <div className="scatter__controls">
        <button className="btn" onClick={reset} title="Reset zoom">
          Reset view
        </button>
      </div>
      {legend && <LegendPanel legend={legend} />}
      <p className="scatter__disclaimer">
        UMAP is a lossy 2D projection. Nearby points are usually related, but chart distance is not
        a faithful similarity measure — the “similar companies” lists are computed in the full
        high-dimensional space.
      </p>
    </div>
  )
}

function LegendPanel({ legend }: { legend: Legend }) {
  return (
    <div className="scatter__legend panel scroll">
      <h3 className="scatter__legendTitle">{legend.label}</h3>
      <ul>
        {legend.entries.map((e) => (
          <li key={e.label}>
            <span className="scatter__swatch" style={{ background: e.color }} aria-hidden="true" />
            <span className="scatter__legendLabel">{e.label}</span>
            <span className="faint mono">{e.count.toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function computeColors(
  dataset: Dataset,
  colorBy: ColorBy,
  activeTagId: string | null,
): { colorIndex: Uint16Array; palette: string[]; legend: Legend | null } {
  const n = dataset.points.ids.length
  const colorIndex = new Uint16Array(n)

  if (colorBy === 'year') {
    const years = dataset.points.year
    const present = [...new Set([...years].filter((y) => y > 0))].sort((a, b) => a - b)
    const min = present[0] ?? 2005
    const max = present[present.length - 1] ?? 2025
    const span = Math.max(1, max - min)
    const counts = new Map<number, number>()
    for (let i = 0; i < n; i += 1) {
      const y = years[i]
      const slot = y > 0
        ? Math.min(YEAR_COLORS.length - 1, Math.round(((y - min) / span) * (YEAR_COLORS.length - 1)))
        : YEAR_COLORS.length - 1
      colorIndex[i] = slot
      counts.set(slot, (counts.get(slot) ?? 0) + 1)
    }
    return {
      colorIndex,
      palette: [...YEAR_COLORS, DIMMED],
      legend: {
        label: 'Batch year',
        entries: YEAR_COLORS.map((color, i) => ({
          color,
          label: String(min + Math.round((i / (YEAR_COLORS.length - 1)) * span)),
          count: counts.get(i) ?? 0,
        })).filter((e) => e.count > 0),
      },
    }
  }

  if (colorBy === 'cluster') {
    const counts = new Map<number, number>()
    for (let i = 0; i < n; i += 1) {
      const c = dataset.points.cluster[i]
      colorIndex[i] = c % CATEGORY_COLORS.length
      counts.set(c, (counts.get(c) ?? 0) + 1)
    }
    return {
      colorIndex,
      palette: [...CATEGORY_COLORS, DIMMED],
      legend: {
        label: 'Algorithmic cluster',
        entries: dataset.clusters.rows
          .slice()
          .sort((a, b) => b.size - a.size)
          .map((c) => ({
            color: CATEGORY_COLORS[c.cluster_id % CATEGORY_COLORS.length],
            label: c.label,
            count: counts.get(c.cluster_id) ?? c.size,
          })),
      },
    }
  }

  if (colorBy === 'tag' && activeTagId) {
    const tag = dataset.tagsById.get(activeTagId)
    let has = 0
    for (let i = 0; i < n; i += 1) {
      const company = dataset.companiesById.get(dataset.points.ids[i])
      const hit = company?.tagIds.includes(activeTagId) ?? false
      colorIndex[i] = hit ? 0 : 1
      if (hit) has += 1
    }
    return {
      colorIndex,
      palette: ['#5ee0c0', '#39414f', DIMMED],
      legend: {
        label: `Tag: ${tag?.name ?? activeTagId}`,
        entries: [
          { color: '#5ee0c0', label: 'assigned', count: has },
          { color: '#39414f', label: 'not assigned', count: n - has },
        ],
      },
    }
  }

  // Default: YC industry.
  const industries = new Map<string, number>()
  const counts = new Map<string, number>()
  for (let i = 0; i < n; i += 1) {
    const company = dataset.companiesById.get(dataset.points.ids[i])
    const key = company?.industry ?? 'Unlisted'
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1])
  // Beyond the palette size the categories become indistinguishable, so the
  // tail is honestly folded into "Other" instead of silently recycling colours.
  const top = ranked.slice(0, CATEGORY_COLORS.length - 1)
  top.forEach(([key], i) => industries.set(key, i))
  const otherIdx = CATEGORY_COLORS.length - 1
  let otherCount = 0
  for (let i = 0; i < n; i += 1) {
    const company = dataset.companiesById.get(dataset.points.ids[i])
    const key = company?.industry ?? 'Unlisted'
    const idx = industries.get(key)
    colorIndex[i] = idx ?? otherIdx
    if (idx === undefined) otherCount += 1
  }
  return {
    colorIndex,
    palette: [...CATEGORY_COLORS, DIMMED],
    legend: {
      label: 'YC industry',
      entries: [
        ...top.map(([label, count], i) => ({ color: CATEGORY_COLORS[i], label, count })),
        ...(otherCount ? [{ color: CATEGORY_COLORS[otherIdx], label: 'Other', count: otherCount }] : []),
      ],
    },
  }
}
