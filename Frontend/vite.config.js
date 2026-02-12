import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // Force all packages to use the SAME single copy of react, react-dom, and three.
    // Without this, nested node_modules (e.g. inside @react-three/fiber or @react-three/drei)
    // can pull in their own bundled copy — causing the "Cannot read properties of undefined"
    // reconciler crash at runtime.
    dedupe: ['react', 'react-dom', 'three'],
    alias: {
      react: path.resolve('./node_modules/react'),
      'react-dom': path.resolve('./node_modules/react-dom'),
    },
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'three',
      '@react-three/fiber',
      '@react-three/drei',
      'react-beautiful-dnd',
    ],
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})