import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Dev: API + WebSocket are proxied to the Node backend on :8088 so the app
// runs same-origin in development. Prod: the backend serves this build, so the
// relative '/api' and '/ws' paths resolve to the same origin automatically.
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg'],
      manifest: {
        name: 'Lean Proof Runner',
        short_name: 'LeanProofs',
        description: 'Browse, visualise, and run Lean 4 / Mathlib proofs from your phone.',
        theme_color: '#0b1021',
        background_color: '#0b1021',
        display: 'standalone',
        orientation: 'portrait',
        icons: [
          { src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any maskable' },
        ],
      },
      workbox: {
        // Cache the file manifest + sources so browsing works offline; runs
        // always hit the network (they need the live backend).
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/api/file'),
            handler: 'StaleWhileRevalidate',
            options: { cacheName: 'lean-files' },
          },
        ],
      },
    }),
  ],
  server: {
    proxy: {
      '/api': 'http://localhost:8088',
      '/ws': { target: 'ws://localhost:8088', ws: true },
    },
  },
});
