import { test, expect } from '@playwright/test';
import { StaticJsonBenchmarkProvider } from '../src/providers/benchmark';

function validBenchmarkData() {
  return {
    schema_version: '1.1.0',
    generated: '2026-09-05T00:00:00Z',
    meta: {
      generator: 'provider-contract-test',
      git_sha: '0'.repeat(40),
      generation_id: '1'.repeat(64),
    },
    environments: [
      {
        env_id: 'test-env',
        label: 'Test environment',
        gpu: 'none',
        cpu: 'test-cpu',
      },
    ],
    categories: [],
    models: [],
    frameworks: [],
    comparisons: [],
    runs: [],
  };
}

test.describe('Benchmark data provider contract', () => {
  test('fails closed for an unsupported benchmark schema', async ({ page }) => {
    await page.route('**/data/benchmark_data.json', async route => {
      const response = await route.fetch();
      const body = await response.json();
      body.schema_version = '99.0.0';
      await route.fulfill({ response, json: body });
    });

    await page.goto('/');
    await expect(page.getByText(/Unsupported schema 99\.0\.0; expected 1\.1\.0/)).toBeVisible();
    await expect(page.locator('.header')).toHaveCount(0);
  });

  test('keeps valid data when optional metadata has another generation', async ({ page }) => {
    await page.route('**/data/parse_report.json', async route => {
      const response = await route.fetch();
      const body = await response.json();
      body.generation_id = '0'.repeat(64);
      await route.fulfill({ response, json: body });
    });

    await page.goto('/');
    await expect(page.locator('.header:not(.header-loading)')).toBeVisible({
      timeout: 60_000,
    });
    await expect(page.locator('.header-meta')).toHaveCount(1);
    await expect(page.locator('#timing-chart')).toBeVisible();
    await expect(page.locator('#env-select option')).toHaveCount(1);
    await expect(page.locator('#env-select')).toHaveValue('remote-p100');
    await expect(page.locator('#env-select option')).toContainText(
      '8 benchmark sessions',
    );
  });

  test('retries a required bundle after a transient failed load', async () => {
    let benchmarkAttempts = 0;
    const fetcher: typeof fetch = async input => {
      const url = String(input);
      if (url.endsWith('/data/benchmark_data.json')) {
        benchmarkAttempts += 1;
        if (benchmarkAttempts === 1) {
          return new Response('temporary failure', { status: 503 });
        }
        return Response.json(validBenchmarkData());
      }
      return new Response('', { status: 404 });
    };

    const provider = new StaticJsonBenchmarkProvider({
      baseUrl: 'https://example.invalid/dashboard/',
      fetcher,
    });

    let firstError: unknown = null;
    try {
      await provider.loadBundle();
    } catch (error) {
      firstError = error;
    }
    expect(String(firstError)).toContain(
      'Failed to load data/benchmark_data.json: 503',
    );

    const recovered = await provider.loadBundle();
    expect(recovered.data.schema_version).toBe('1.1.0');
    expect(recovered.data.environments[0].env_id).toBe('test-env');
    expect(benchmarkAttempts).toBe(2);

    // A successful bundle remains cached; only the rejected request is evicted.
    await provider.loadBundle();
    expect(benchmarkAttempts).toBe(2);
  });

  test('ignores malformed optional metadata instead of rejecting valid data', async () => {
    const fetcher: typeof fetch = async input => {
      const url = String(input);
      if (url.endsWith('/data/benchmark_data.json')) {
        return Response.json(validBenchmarkData());
      }
      if (
        url.endsWith('/data/parse_report.json') ||
        url.endsWith('/data/source_inventory.json')
      ) {
        return new Response('{not valid json', {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }
      return new Response('', { status: 404 });
    };

    const provider = new StaticJsonBenchmarkProvider({
      baseUrl: 'https://example.invalid/dashboard/',
      fetcher,
    });
    const bundle = await provider.loadBundle();

    expect(bundle.data.schema_version).toBe('1.1.0');
    expect(bundle.parseReport).toBeNull();
    expect(bundle.sourceInventory).toBeNull();
  });
});
