import { test } from '@playwright/test';
import { assertConvergenceSessionProvenance } from './helpers/convergence-provenance-browser';

test('convergence rows retain source identities through pagination and filters', async ({ page }) => {
  await page.goto('/');
  await assertConvergenceSessionProvenance(page);
});
