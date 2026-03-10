const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  publicPath: '/',
  productionSourceMap: false,
  devServer: {
    proxy: {
      '/api': { target: 'http://localhost:8080', changeOrigin: true },
      '/status': { target: 'http://localhost:8080', changeOrigin: true },
      '/predict': { target: 'http://localhost:8080', changeOrigin: true },
      '/split': { target: 'http://localhost:8080', changeOrigin: true },
      '/extract': { target: 'http://localhost:8080', changeOrigin: true },
      '/process': { target: 'http://localhost:8080', changeOrigin: true },
      '/convert': { target: 'http://localhost:8080', changeOrigin: true },
      '/generate': { target: 'http://localhost:8080', changeOrigin: true },
      '/sort': { target: 'http://localhost:8080', changeOrigin: true },
      '/job': { target: 'http://localhost:8080', changeOrigin: true },
      '/download': { target: 'http://localhost:8080', changeOrigin: true },
      '/upload': { target: 'http://localhost:8080', changeOrigin: true },
      '/manual': { target: 'http://localhost:8080', changeOrigin: true },
      '/health': { target: 'http://localhost:8080', changeOrigin: true },
      '/merge': { target: 'http://localhost:8080', changeOrigin: true },
      '/check': { target: 'http://localhost:8080', changeOrigin: true },
      '/refine': { target: 'http://localhost:8080', changeOrigin: true },
    }
  }
})
