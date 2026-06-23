import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // /api 는 Django(8000)로 프록시 → 단일 origin, CORS 우회.
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
