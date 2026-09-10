import { expect, test } from '@playwright/test';

test('renders inline and display LaTeX on documentation pages', async ({ page }) => {
  await page.goto('/statgpu/en/panel/random-effects');

  const equations = page.locator('mjx-container');
  await expect(equations.first()).toBeVisible({ timeout: 60_000 });
  expect(await equations.count()).toBeGreaterThan(5);
  expect(await page.locator('mjx-container[display="true"]').count()).toBeGreaterThan(0);
  await expect(page.locator('main')).not.toContainText('$$');

  await page.goto('/statgpu/cn/models/linear-regression');
  const explanation = page.locator('main').getByText(
    /是观测数.*是第.*个观测的特征向量/,
  );
  await expect(explanation).toBeVisible();
  expect(await explanation.locator('mjx-container').count()).toBeGreaterThanOrEqual(4);
  await expect(explanation).not.toContainText('\\varepsilon_i');

  await page.goto('/statgpu/en/models/glm-family-reference');
  const covarianceSection = page.locator(
    '#covariance-by-family-link-and-covariance-type',
  );
  await expect(covarianceSection).toBeVisible();
  expect(await page.locator('main mjx-container').count()).toBeGreaterThan(10);
  await expect(page.locator('main')).not.toContainText('$$');
});

test('renders every general solver algorithm with MathJax in both languages', async ({
  page,
}) => {
  const pages = [
    {
      route: '/statgpu/en/guides/solver-algorithms',
      firstAlgorithm: '1. FISTA',
      finalAlgorithm: '10. Exact Ridge solve',
    },
    {
      route: '/statgpu/cn/guides/solver-algorithms',
      firstAlgorithm: '1. FISTA',
      finalAlgorithm: '10. Exact Ridge',
    },
  ];

  for (const solverPage of pages) {
    await page.goto(solverPage.route);
    const main = page.getByRole('main');
    await expect(
      main.getByRole('heading', { name: solverPage.firstAlgorithm }),
    ).toBeVisible();
    await expect(
      main.getByRole('heading', { name: solverPage.finalAlgorithm }),
    ).toBeVisible();

    const numberedSections = main.locator('h2').filter({ hasText: /^\d+\./ });
    await expect(numberedSections).toHaveCount(10);
    expect(
      await main.locator('mjx-container[display="true"]').count(),
    ).toBeGreaterThanOrEqual(20);
    await expect(main).not.toContainText('$$');
    await expect(main).not.toContainText('Quantile coordinate descent');
  }

  // Quantile coordinate descent is intentionally model-specific rather than a
  // general solver path. Verify its formulas on the model page instead of
  // reintroducing it into the shared solver inventory.
  await page.goto('/statgpu/en/models/quantile');
  const quantileMain = page.getByRole('main');
  await expect(
    quantileMain.getByRole('heading', { name: 'Quantile coordinate descent' }),
  ).toBeVisible();
  expect(
    await quantileMain.locator('mjx-container[display="true"]').count(),
  ).toBeGreaterThan(0);
  await expect(quantileMain).not.toContainText('$$');
});
