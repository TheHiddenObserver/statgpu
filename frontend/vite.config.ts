import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: '../docs/assets/benchmarks',
    emptyOutDir: true,
  },
});
