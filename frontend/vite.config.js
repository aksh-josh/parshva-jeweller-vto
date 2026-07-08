import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Falls back to 127.0.0.1:5000 for local `npm run dev` outside Docker;
// docker-compose.yml sets VITE_PROXY_TARGET=http://backend:5000 so the
// proxy reaches the right container when running inside Docker.
const apiTarget = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:5000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,       // lets Vite's dev server be reached from outside
                       // the container (Vite defaults to localhost-only)
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        secure: false,
      },
      '/static': {
        target: apiTarget,
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
