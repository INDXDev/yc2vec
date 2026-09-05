/**
 * Browser checks against a running build.
 *
 *   npm run build && npm run preview &
 *   node scripts/browser-checks.mjs http://localhost:4173/
 *   node scripts/browser-checks.mjs https://indxdev.github.io/yc2vec/
 *
 * Three things are verified, all of which are invisible to unit tests:
 *
 *  1. **Accessibility basics.** Landmarks, a skip link, `lang`, live regions,
 *     an accessible name on every control (including implicit `<label>`
 *     wrapping), alt text on images, and no skipped heading levels.
 *  2. **Deep-link routing under a subpath.** GitHub Pages has no rewrite rules,
 *     so a route that 404s on refresh is a real deployment bug. Every route is
 *     loaded cold and then hard-refreshed.
 *  3. **Runtime errors.** The WebGL scatterplot fails silently if it fails at
 *     all, so console and page errors are collected rather than assumed absent.
 *
 * Requires a Chromium that playwright-core can drive. Set CHROME_PATH to point
 * at one, or install with `npx playwright install chromium`.
 */

import { chromium } from 'playwright-core'

const base = (process.argv[2] ?? 'http://localhost:4173/').replace(/\/?$/, '/')
const executablePath = process.env.CHROME_PATH || undefined

const ROUTES = [
  ['', 'DNA map', null],
  ['#/search', 'Search explorer', 'All companies'],
  ['#/tags', 'Tag explorer', 'Tag explorer'],
  ['#/about', 'About', 'How YC2Vec is built'],
  ['#/search?q=data', 'Search with a query', null],
]

const failures = []
const note = (ok, label, detail = '') => {
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures.push(`${label}${detail ? `: ${detail}` : ''}`)
}

const browser = await chromium.launch({
  executablePath,
  // Software GL: CI runners have no GPU, and regl needs a working context.
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
})

console.log(`\nchecking ${base}\n`)

// -- 1. accessibility ---------------------------------------------------------
console.log('accessibility')
for (const [hash, label] of ROUTES.slice(0, 4)) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  await page.goto(base + hash, { waitUntil: 'networkidle' })
  await page.waitForTimeout(1500)

  const report = await page.evaluate(() => {
    const problems = []
    for (const img of document.querySelectorAll('img')) {
      if (!img.hasAttribute('alt')) problems.push(`img without alt: ${img.src.slice(0, 60)}`)
    }
    for (const el of document.querySelectorAll('button, a, input, select')) {
      // An accessible name may come from aria-label, own text, label[for],
      // an ancestor <label>, title or placeholder.
      const name =
        (el.getAttribute('aria-label') || el.textContent || '').trim() ||
        (el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)?.textContent?.trim()) ||
        el.closest('label')?.textContent?.trim() ||
        el.getAttribute('title') ||
        el.getAttribute('placeholder')
      if (!name) problems.push(`unnamed ${el.tagName.toLowerCase()}: ${el.outerHTML.slice(0, 60)}`)
    }
    const levels = [...document.querySelectorAll('h1,h2,h3,h4')].map((h) => +h.tagName[1])
    for (let i = 1; i < levels.length; i += 1) {
      if (levels[i] - levels[i - 1] > 1) problems.push(`heading jump h${levels[i - 1]} → h${levels[i]}`)
    }
    return {
      problems,
      main: !!document.querySelector('main'),
      nav: !!document.querySelector('nav'),
      skipLink: !!document.querySelector('.skip-link'),
      live: document.querySelectorAll('[aria-live]').length,
      lang: document.documentElement.lang,
    }
  })

  note(report.problems.length === 0, `${label}: control names, alt text, heading order`,
       report.problems.slice(0, 3).join('; '))
  note(report.main && report.nav && report.skipLink, `${label}: landmarks and skip link`)
  note(report.live > 0, `${label}: has a live region for result counts`)
  note(report.lang === 'en', `${label}: document language`, report.lang)
  await page.close()
}

// -- 2. deep links survive a refresh under the subpath ------------------------
console.log('\ndeep links (cold load + hard refresh)')
for (const [hash, label, expected] of ROUTES) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  const status = (await page.goto(base + hash, { waitUntil: 'networkidle' }))?.status()
  await page.waitForTimeout(1500)
  await page.reload({ waitUntil: 'networkidle' })
  await page.waitForTimeout(1500)
  const after = await page.evaluate(() => ({ text: document.body.innerText, hash: location.hash }))
  const ok = status === 200 && after.hash === hash && (expected ? after.text.includes(expected) : after.text.length > 200)
  note(ok, `${label} (${hash || '/'})`, ok ? '' : `http=${status} hash=${after.hash}`)
  await page.close()
}

// -- 3. no runtime errors, and the map actually paints ------------------------
console.log('\nruntime')
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  const errors = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(String(e.message)))
  await page.goto(base, { waitUntil: 'networkidle' })
  await page.waitForTimeout(3000)
  const state = await page.evaluate(() => ({
    canvas: document.querySelectorAll('canvas').length,
    count: document.querySelector('.app__count')?.textContent ?? '',
    legend: !!document.querySelector('.scatter__legendTitle'),
    disclaimer: document.querySelector('.scatter__disclaimer')?.textContent ?? '',
  }))
  note(errors.length === 0, 'no console or page errors', errors.slice(0, 2).join(' | '))
  note(state.canvas === 1, 'scatterplot canvas is present')
  note(state.legend, 'legend rendered')
  note(/\d/.test(state.count), 'company count rendered', state.count)
  note(state.disclaimer.toLowerCase().includes('lossy'), 'projection caveat is visible to the reader')
  await page.close()
}

// -- 4. both themes render with an explicit background ------------------------
console.log('\nthemes')
for (const colorScheme of ['dark', 'light']) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, colorScheme, reducedMotion: 'reduce' })
  await page.goto(base, { waitUntil: 'networkidle' })
  await page.waitForTimeout(1200)
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
  // A transparent body would inherit whatever the host paints behind it.
  note(bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent', `${colorScheme} theme paints a background`, bg)
  await page.close()
}

await browser.close()

console.log('')
if (failures.length) {
  console.error(`${failures.length} check(s) failed:\n  - ${failures.join('\n  - ')}`)
  process.exit(1)
}
console.log('all browser checks passed')
