import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { dashboardURL } from './e2e-production/site';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const productionURL = dashboardURL;

export default defineConfig({
  testDir: './e2e-production',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 90_000,
  reporter: 'html',
  use: {
    baseURL: productionURL,
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium-production', use: { ...devices['Desktop Chrome'] } },
    {
      name: 'firefox-production',
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: {
          // Playwright's Firefox strategy 0 duplicates history entries on reload:
          // https://github.com/microsoft/playwright/issues/22640
          // https://bugzilla.mozilla.org/show_bug.cgi?id=1832341
          // These same-origin production checks do not exercise OOPIFs, which
          // the Firefox driver does not fully support with strategy 1.
          firefoxUserPrefs: { 'fission.webContentIsolationStrategy': 1 },
        },
      },
    },
    { name: 'webkit-production', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'node node_modules/vitepress/bin/vitepress.js preview docs --host 127.0.0.1 --port 4173',
    url: productionURL,
    reuseExistingServer: !process.env.CI,
    cwd: dirname(__dirname),
    timeout: 180_000,
  },
});
