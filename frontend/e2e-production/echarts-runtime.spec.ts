import { test, expect } from '@playwright/test';

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
});
