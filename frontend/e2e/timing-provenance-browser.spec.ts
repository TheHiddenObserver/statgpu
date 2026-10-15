import { test } from '@playwright/test';
import { assertTimingSessionProvenance } from './helpers/timing-provenance-browser';

test('grouped timing charts expose CV provenance in tooltips and keyboard-accessible tables', async ({ page }) => {
  await page.goto('/');
  await assertTimingSessionProvenance(page);
});
