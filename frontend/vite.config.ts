import { defineConfig } from 'vite';

export default defineConfig({
  // The dashboard artifact is relocatable. It is assembled under
  // <site-base>/dashboard/ for both project Pages and custom domains.
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
