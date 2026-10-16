import { test } from '@playwright/test';
import { assertConvergenceSessionProvenance } from '../e2e/helpers/convergence-provenance-browser';

test('deployed convergence rows retain source identities through pagination and filters', async ({ page }) => {
  await page.goto('./');
  await assertConvergenceSessionProvenance(page);
});
