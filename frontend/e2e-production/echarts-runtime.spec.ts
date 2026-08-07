import { test, expect, type Locator } from '@playwright/test';

async function announcedRowCount(details: Locator): Promise<number> {
  const text = (await details.locator('summary').textContent()) ?? '';
  const match = text.match(/\((\d+) rows\)$/);
  expect(match, `expected row count in disclosure summary: ${text}`).not.toBeNull();
  return Number(match![1]);
}

test.describe('Deployed benchmark chart runtime', () => {
  test('completes tree-shaken ECharts initialization without async runtime errors', async ({ page }) => {
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];
    page.on('console', message => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', error => pageErrors.push(error.message));

    await page.goto('./');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });

    // Chart rendering is intentionally deferred with requestAnimationFrame in
    // main.ts. Wait for both renderers to enter the real setOption path and
    // produce their CanvasRenderer surfaces before checking async errors.
    await page.waitForFunction(() => {
      const timing = document.querySelector<HTMLElement>('#timing-chart');
      const speedup = document.querySelector<HTMLElement>('#speedup-chart');
      return Boolean(
        timing?.dataset.timingDisplayed !== undefined &&
        speedup?.dataset.speedupDisplayed !== undefined &&
        timing.querySelector('canvas') &&
        speedup.querySelector('canvas'),
      );
    });
    await page.evaluate(
      () => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))),
    );

    const runtime = await page.evaluate(() => {
      const timing = document.querySelector<HTMLElement>('#timing-chart')!;
      const speedup = document.querySelector<HTMLElement>('#speedup-chart')!;
      const timingCanvas = timing.querySelector<HTMLCanvasElement>('canvas')!;
      const speedupCanvas = speedup.querySelector<HTMLCanvasElement>('canvas')!;
      return {
        timingDisplayed: Number(timing.dataset.timingDisplayed ?? 0),
        speedupDisplayed: Number(speedup.dataset.speedupDisplayed ?? 0),
        timingCanvas: [timingCanvas.width, timingCanvas.height],
        speedupCanvas: [speedupCanvas.width, speedupCanvas.height],
      };
    });

    expect(runtime.timingDisplayed).toBeGreaterThan(0);
    expect(runtime.speedupDisplayed).toBeGreaterThan(0);
    expect(runtime.timingCanvas[0]).toBeGreaterThan(0);
    expect(runtime.timingCanvas[1]).toBeGreaterThan(0);
    expect(runtime.speedupCanvas[0]).toBeGreaterThan(0);
    expect(runtime.speedupCanvas[1]).toBeGreaterThan(0);
    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });

  test('materializes exactly the announced chart rows and resets them on rerender', async ({ page }) => {
    await page.goto('./');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });

    const timing = page.locator('#timing-chart-data');
    const speedup = page.locator('#speedup-chart-data');
    const focusedTimingRows = await announcedRowCount(timing);
    const focusedSpeedupRows = await announcedRowCount(speedup);

    await expect(timing.locator('table')).toHaveCount(0);
    await expect(speedup.locator('table')).toHaveCount(0);
    await timing.locator('summary').click();
    await speedup.locator('summary').click();
    await expect(timing.locator('tbody tr')).toHaveCount(focusedTimingRows);
    await expect(speedup.locator('tbody tr')).toHaveCount(focusedSpeedupRows);

    // A filter/view update replaces the dashboard main content. The newly
    // selected disclosure must start unmaterialized and derive rows from the
    // new selection rather than retaining the previous table DOM.
    await page.locator('[data-chart-view="full"]').click();
    await expect(timing.locator('table')).toHaveCount(0);
    await expect(speedup.locator('table')).toHaveCount(0);

    const fullTimingRows = await announcedRowCount(timing);
    const fullSpeedupRows = await announcedRowCount(speedup);
    expect(fullTimingRows).toBeGreaterThanOrEqual(focusedTimingRows);
    expect(fullSpeedupRows).toBeGreaterThanOrEqual(focusedSpeedupRows);

    await timing.locator('summary').click();
    await speedup.locator('summary').click();
    await expect(timing.locator('tbody tr')).toHaveCount(fullTimingRows);
    await expect(speedup.locator('tbody tr')).toHaveCount(fullSpeedupRows);
  });
});
