import { defineConfig } from 'vite';

export default defineConfig({
  // Keep the dashboard relocatable beneath <site-base>/dashboard/.
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
