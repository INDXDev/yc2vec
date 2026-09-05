import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider, createHashRouter } from 'react-router-dom'
import App from './App'
import { DnaMapView } from './views/DnaMapView'
import { CompanyView } from './views/CompanyView'
import { TagExplorerView } from './views/TagExplorerView'
import { SearchExplorerView } from './views/SearchExplorerView'
import { AboutView } from './views/AboutView'
import './styles/global.css'

/**
 * Hash routing.
 *
 * GitHub Pages has no rewrite rules, so a browser-history route like
 * `/yc2vec/company/ycoss:5` 404s on a direct refresh. Hash routes are served by
 * `index.html` at every depth, which makes deep links and refresh work
 * identically under a project subpath and on a custom domain.
 */
const router = createHashRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <DnaMapView /> },
      { path: 'company/:companyId', element: <CompanyView /> },
      { path: 'tags', element: <TagExplorerView /> },
      { path: 'tags/:tagId', element: <TagExplorerView /> },
      { path: 'search', element: <SearchExplorerView /> },
      { path: 'about', element: <AboutView /> },
    ],
  },
])

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
