import { expect, type Page } from '@playwright/test';
import type { BenchmarkData } from '../../src/schema';

/** Shared dev and deployed-browser assertion against the served canonical data. */
export async function assertTimingSessionProvenance(page: Page): Promise<void> {
  await expect(page.locator('.header:not(.header-loading)')).toBeVisible({ timeout: 60_000 });
  const response = await page.request.get(new URL('data/benchmark_data.json', page.url()).toString());
  expect(response.ok()).toBe(true);
  const data: BenchmarkData = await response.json();
  const sources = ['cv_benchmark_20260807.json', 'cv_benchmark_pr116_p100.json'];
  const runs = data.runs.filter(run => run.model_id === 'RidgeCV' && sources.includes(run.source.file));
  expect(runs).toHaveLength(6);
  await expect(page.locator('#env-select option')).toHaveCount(1);
  await expect(page.locator('#env-select')).toHaveValue('remote-p100');
  await page.getByRole('button', { name: 'None', exact: true }).click();
  await page.locator('#cat-linear_models').check();
  await page.locator('[data-metric-scope="cross_validation"]').click();
  await page.getByLabel('Model', { exact: true }).selectOption('RidgeCV');

  for (const view of ['focused', 'full']) {
    await page.locator(`button[data-chart-view="${view}"]`).click();
    const chart = page.locator('#timing-chart');
    await expect(chart).toHaveAttribute('data-chart-view', view);
    await expect(chart).toHaveAttribute('data-timing-displayed', '6');
    await expect(chart).toHaveAttribute('aria-describedby', 'timing-chart-data');
    await expect(chart.locator('canvas')).toBeVisible();

    const details = page.locator('#timing-chart-data');
    await expect(details.locator('table')).toHaveCount(0);
    await details.locator('summary').focus();
    await details.locator('summary').press('Enter');
    await expect(details.locator('tbody tr')).toHaveCount(6);
    await expect(details.getByRole('columnheader', { name: 'Benchmark session', exact: true })).toHaveAttribute('scope', 'col');
    await expect(details.getByRole('columnheader', { name: 'Source', exact: true })).toHaveAttribute('scope', 'col');
    for (const run of runs) {
      const row = details.locator(`tr[data-run-id="${run.run_id}"]`);
      await expect(row).toContainText(run.benchmark_session_id ?? run.env_id);
      await expect(row).toContainText(run.source.file);
      await expect(row).toContainText(`${run.backend}/${run.implementation}`);
      await expect(row).toContainText(`${run.metrics.timing!.fit_time_ms.toFixed(3)} ms`);
    }

    // Hover the actual ECharts canvas. Axis-triggered tooltips need no private
    // chart instance, bundle-export knowledge, or development-only test API.
    await chart.scrollIntoViewIfNeeded();
    const bounds = await chart.boundingBox();
    expect(bounds).not.toBeNull();
    const observed = new Set<string>();
    for (let step = 0; step < 24; step += 1) {
      await chart.hover({ position: { x: 28 + (bounds!.width - 48) * (step + 0.5) / 24, y: 110 } });
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())));
      const tooltip = chart.locator('.timing-tooltip:visible');
      if (await tooltip.count() === 0) continue;
      const text = (await tooltip.textContent()) ?? '';
      for (const run of runs) {
        if (text.includes(run.source.file) && text.includes(`${run.backend}/${run.implementation}:`)) {
          expect(text).toContain(`Session: ${run.benchmark_session_id ?? run.env_id}`);
          expect(text).toContain(`${run.metrics.timing!.fit_time_ms.toFixed(2)} ms`);
          expect(text).not.toContain(sources.find(source => source !== run.source.file)!);
          observed.add(run.run_id);
        }
      }
    }
    expect([...observed].sort(), `${view} tooltips must identify all six source/backend rows`).toEqual(runs.map(run => run.run_id).sort());
  }
}
