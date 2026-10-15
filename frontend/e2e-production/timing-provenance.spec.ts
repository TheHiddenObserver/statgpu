import { test } from '@playwright/test';
import { assertTimingSessionProvenance } from '../e2e/helpers/timing-provenance-browser';

test('deployed grouped timing charts retain both CV sources and accessible identities', async ({ page }) => {
  await page.goto('./');
  await assertTimingSessionProvenance(page);
});
