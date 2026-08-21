import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

const DEFAULT_API_PORT = '8000'

export default defineConfig(({ mode }) => {
  const projectEnvironment = loadEnv(mode, '..', '')
  const apiPort = projectEnvironment.STKB_API_PORT || DEFAULT_API_PORT

  return {
    plugins: [vue()],
    server: {
      proxy: {
        '/api': {
          target: `http://127.0.0.1:${apiPort}`,
          changeOrigin: true,
        },
      },
    },
  }
})
