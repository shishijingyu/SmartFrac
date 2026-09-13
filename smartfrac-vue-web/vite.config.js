import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 91,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:48091',
        changeOrigin: true
      }
    }
  }
})
