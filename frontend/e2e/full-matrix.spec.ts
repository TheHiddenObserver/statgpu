import { expect, test } from '@playwright/test';

test('Full matrix exposes every timing group and speedup row', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });

  const timingChart = page.locator('#timing-chart');
  const speedupChart = page.locator('#speedup-chart');
  await page.getByRole('button', { name: 'Full matrix' }).click();

  await expect(timingChart).toHaveAttribute('data-chart-view', 'full');
  await expect(speedupChart).toHaveAttribute('data-chart-view', 'full');

  const totalTimingGroups = Number(
    await timingChart.getAttribute('data-timing-groups'),
  );
  const displayedTimingGroups = Number(
    await timingChart.getAttribute('data-timing-displayed'),
  );
  expect(totalTimingGroups).toBeGreaterThan(0);
  expect(displayedTimingGroups).toBe(totalTimingGroups);

  const totalSpeedupRows = Number(
    await speedupChart.getAttribute('data-speedup-rows'),
  );
  const displayedSpeedupRows = Number(
    await speedupChart.getAttribute('data-speedup-displayed'),
  );
  expect(totalSpeedupRows).toBeGreaterThan(0);
  expect(displayedSpeedupRows).toBe(totalSpeedupRows);
});
