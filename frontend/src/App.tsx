import { NavLink, Outlet, useSearchParams } from 'react-router-dom'
import { DatasetProvider, useDataset } from './lib/DatasetContext'
import { SearchBox } from './components/SearchBox'
import './styles/app.css'

const NAV = [
  { to: '/', label: 'DNA map', end: true },
  { to: '/search', label: 'Search' },
  { to: '/tags', label: 'Tags' },
  { to: '/about', label: 'About' },
]

function Header() {
  const { dataset, results, loading } = useDataset()
  const [params] = useSearchParams()
  const suffix = params.toString() ? `?${params.toString()}` : ''

  return (
    <header className="app__header">
      <div className="app__brand">
        <span className="app__logo" aria-hidden="true" />
        <div>
          <span className="app__title">YC2Vec</span>
          <span className="app__tagline">semantic DNA of Y Combinator</span>
        </div>
      </div>

      <nav className="app__nav" aria-label="Primary">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={`${item.to}${suffix}`}
            end={item.end}
            className={({ isActive }) => `app__navlink${isActive ? ' is-active' : ''}`}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="app__headerRight">
        <SearchBox />
        <span className="app__count mono" aria-live="polite">
          {loading
            ? 'loading…'
            : `${results.length.toLocaleString()} / ${(dataset?.companies.length ?? 0).toLocaleString()}`}
        </span>
      </div>
    </header>
  )
}

function Shell() {
  const { error } = useDataset()

  if (error) {
    return (
      <div className="app">
        <Header />
        <main className="app__error" id="main">
          <h1>The dataset could not be loaded</h1>
          <p className="muted">{error.message}</p>
          <p className="faint">
            YC2Vec is a static site: it downloads a precomputed dataset and then runs entirely in
            your browser. If you are offline, reconnect and reload.
          </p>
          <button className="btn btn--primary" onClick={() => window.location.reload()}>
            Retry
          </button>
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <Header />
      <main className="app__main" id="main">
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <DatasetProvider>
      <Shell />
    </DatasetProvider>
  )
}
