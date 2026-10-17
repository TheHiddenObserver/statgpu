import { expect, type Page } from '@playwright/test';
import type { BenchmarkData, Run } from '../../src/schema';

/** Shared dev and deployed-browser check using the actual served canonical data. */
export async function assertConvergenceSessionProvenance(page: Page): Promise<void> {
  await expect(page.locator('.header:not(.header-loading)')).toBeVisible({ timeout: 60_000 });
  const response = await page.request.get(new URL('data/benchmark_data.json', page.url()).toString());
  expect(response.ok()).toBe(true);
  const data: BenchmarkData = await response.json();
  const runs = data.runs.filter(run =>
    ['cv_benchmark_20260807.json', 'cv_benchmark_pr116_p100.json'].includes(run.source.file)
    && run.metrics.convergence
    && (run.metrics.convergence.n_iter_mean != null || run.metrics.convergence.converged_rate != null),
  );
  expect(runs).toHaveLength(43);
  await expect(page.locator('#env-select')).toHaveValue('remote-p100');
  await page.locator('.sidebar-actions').getByRole('button', { name: 'All', exact: true }).click();
  await page.locator('[data-metric-scope="cross_validation"]').click();
  const toggle = page.getByRole('button', { name: /Convergence Stats/ });
  const panel = page.locator('#convergence-panel-body');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await toggle.focus();
  await toggle.press('Enter');
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(panel.getByRole('columnheader')).toHaveText([
    'Model', 'Scale', 'Backend', 'Benchmark session / 基准会话', 'Source / 来源',
    'N Iter (mean±std)', 'Converged Rate',
  ]);
  await expect(panel.getByRole('columnheader', { name: 'Benchmark session / 基准会话', exact: true }))
    .toHaveAttribute('scope', 'col');
  await expect(panel.getByRole('columnheader', { name: 'Source / 来源', exact: true }))
    .toHaveAttribute('scope', 'col');

  const assertRows = async (expected: Run[]) => {
    const expectedCells = expected.map(run => {
      const metric = run.metrics.convergence!;
      return [
        run.model_id, run.scale.label, run.backend ?? run.framework,
        run.benchmark_session_id ?? run.env_id, run.source.file,
        metric.n_iter_mean != null
          ? `${metric.n_iter_mean.toFixed(1)}${metric.n_iter_std != null ? ` ± ${metric.n_iter_std.toFixed(1)}` : ''}`
          : '-',
        metric.converged_rate != null ? `${(metric.converged_rate * 100).toFixed(0)}%` : '-',
      ];
    });
    await expect.poll(() => panel.locator('tr').evaluateAll(rows => rows
      .filter(row => row.querySelector('td'))
      .map(row => [...row.querySelectorAll('td')].map(cell => cell.textContent)),
    )).toEqual(expectedCells);
  };

  // The default page contains 35 statgpu rows, so provenance must survive pagination.
  const statgpuRuns = runs.filter(run => run.framework === 'statgpu');
  expect(statgpuRuns).toHaveLength(35);
  await assertRows(statgpuRuns.slice(0, 30));
  await panel.getByRole('button', { name: 'Show all 35', exact: true }).click();
  await assertRows(statgpuRuns);
  await page.locator('input[type="checkbox"][value="sklearn"]').check();
  await assertRows(runs);

  // Chart view and overview sort must not reorder or collapse metric-panel rows.
  for (const view of ['full', 'focused']) {
    await page.locator(`button[data-chart-view="${view}"]`).click();
    await assertRows(runs);
  }
  await page.locator('.table-container > table').getByRole('columnheader', { name: 'Time (ms)' }).click();
  await assertRows(runs);
  await panel.getByRole('button', { name: 'Show first 30', exact: true }).click();
  await assertRows(runs.slice(0, 30));
  await panel.getByRole('button', { name: 'Show all 43', exact: true }).click();
  await assertRows(runs);
  await toggle.press('Enter');
  await expect(panel).toHaveCount(0);
  await toggle.press('Enter');
  await assertRows(runs);

  // A model change resets the external-framework filter; backend filtering keeps
  // both benchmark sessions and leaves explicitly selected references visible.
  await page.getByLabel('Model', { exact: true }).selectOption('RidgeCV');
  const ridgeRuns = runs.filter(run => run.model_id === 'RidgeCV');
  await assertRows(ridgeRuns.filter(run => run.framework === 'statgpu'));
  await page.locator('input[type="checkbox"][value="sklearn"]').check();
  await assertRows(ridgeRuns);
  await page.locator('input[name="backend"][value="numpy"]').check();
  await assertRows(ridgeRuns.filter(run => run.backend === 'numpy' || run.framework === 'sklearn'));
  await page.locator('input[type="checkbox"][value="sklearn"]').uncheck();
  await assertRows(ridgeRuns.filter(run => run.backend === 'numpy'));
}
