
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

function loadLocalAdminKey(): string {
  const envPath = resolve(__dirname, '../.env')
  const lines = readFileSync(envPath, 'utf8').split(/\r?\n/)

  const entry = lines.find((line) =>
    /^\s*ADMIN_API_KEY\s*=/.test(line)
  )

  if (!entry) {
    throw new Error('ADMIN_API_KEY is missing from the project .env')
  }

  let value = entry.split('=').slice(1).join('=').trim()

  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    value = value.slice(1, -1)
  }

  if (!value) {
    throw new Error('ADMIN_API_KEY is empty')
  }

  return value
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api/demo': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader(
              'X-MedOnto-Admin-Key',
              loadLocalAdminKey(),
            )
          })
        },
      },
    },
  },
})
