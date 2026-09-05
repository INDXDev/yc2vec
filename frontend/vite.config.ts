import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site under /<repo>/. The base path is injected
// at build time so the same source works on Pages, on a custom domain (base
// '/'), and in local dev.
const base = process.env.VITE_BASE_PATH ?? '/'

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // Keep the scatterplot renderer and the search engine out of the entry
        // chunk so the shell paints before either is needed.
        manualChunks: {
          scatter: ['regl-scatterplot'],
          search: ['minisearch'],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
    css: false,
  },
} as never)
