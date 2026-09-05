import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import App from '../App'

/**
 * The degraded state.
 *
 * A static site that cannot reach its dataset has nothing to show, which makes
 * it exactly the moment the shell has to stay usable and honest: say what went
 * wrong, keep the navigation and the skip link, and offer a way out. CI caught
 * a missing skip link here by accidentally running the browser checks against
 * an app with no data, which is a good argument for testing it deliberately.
 */

function renderApp() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route path="/*" element={<App />} />
      </Routes>
    </MemoryRouter>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('when the dataset cannot be loaded', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 500 })))
  })

  it('explains what happened instead of showing an empty page', async () => {
    renderApp()
    expect(await screen.findByRole('heading', { name: /could not be loaded/i })).toBeInTheDocument()
    expect(screen.getByText(/static site/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('keeps the landmarks and the skip link', async () => {
    renderApp()
    await screen.findByRole('heading', { name: /could not be loaded/i })
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeInTheDocument()
    const skip = screen.getByRole('link', { name: 'Skip to main content' })
    expect(skip).toHaveAttribute('href', '#main')
  })
})

describe('when the browser is offline', () => {
  beforeEach(() => {
    // navigator is not configurable via stubGlobal in jsdom; onLine has to be
    // redefined on the existing object.
    Object.defineProperty(window.navigator, 'onLine', { value: false, configurable: true })
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))
  })

  afterEach(() => {
    Object.defineProperty(window.navigator, 'onLine', { value: true, configurable: true })
  })

  it('says so rather than blaming the dataset', async () => {
    renderApp()
    // The specific message, not just the word: the generic copy also mentions
    // being offline, so a loose matcher would pass without the detection.
    expect(
      await screen.findByText('You appear to be offline. YC2Vec needs to download its dataset once.'),
    ).toBeInTheDocument()
  })
})
