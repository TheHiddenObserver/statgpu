import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const productionURL = 'http://127.0.0.1:4173/docs/assets/benchmarks/';

export default defineConfig({
  testDir: './e2e-production',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: productionURL,
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium-production', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox-production', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit-production', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'python3 -m http.server 4173 --bind 127.0.0.1 --directory ..',
    url: `${productionURL}index.html`,
    reuseExistingServer: !process.env.CI,
    cwd: __dirname,
  },
});
