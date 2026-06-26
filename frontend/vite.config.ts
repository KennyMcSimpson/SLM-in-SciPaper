import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), basicSsl()],

  // ─── Dev Proxy ─────────────────────────────────────────────────────────────
  // Forwards /api/* to the Flask backend during local development.
  // In production, Nginx handles this routing transparently.
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },

  // ─── Worker Config ─────────────────────────────────────────────────────────
  // Ensure Web Workers are bundled correctly with Rollup.
  // The ?worker import suffix is used in Vue components.
  worker: {
    format: 'es',
  },

  // ─── Build Optimisation ────────────────────────────────────────────────────
  // Split large ONNX Runtime vendor chunk separately to avoid timeouts.
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('onnxruntime-web')) return 'onnxruntime'
          if (id.includes('pdfjs-dist')) return 'pdfjs'
          if (id.includes('node_modules')) return 'vendor'
        },
      },
    },
    // Increase chunk size warning threshold (ONNX models are large)
    chunkSizeWarningLimit: 2048,
  },

  // ─── WASM MIME type ────────────────────────────────────────────────────────
  // Vite's dev server must serve .wasm files with the correct MIME type for
  // ONNX Runtime Web's WebAssembly backend to initialise.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
