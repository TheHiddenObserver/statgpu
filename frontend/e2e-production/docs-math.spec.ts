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

test('renders every solver algorithm with MathJax in both languages', async ({
  page,
}) => {
  const pages = [
    {
      route: '/statgpu/en/guides/solver-algorithms',
      missingAlgorithm: '8. Quantile coordinate descent',
      finalAlgorithm: '12. Exact Ridge solve',
    },
    {
      route: '/statgpu/cn/guides/solver-algorithms',
      missingAlgorithm: '8. Quantile coordinate descent',
      finalAlgorithm: '12. Exact Ridge',
    },
  ];

  for (const solverPage of pages) {
    await page.goto(solverPage.route);
    const main = page.getByRole('main');
    await expect(
      main.getByRole('heading', { name: solverPage.missingAlgorithm }),
    ).toBeVisible();
    await expect(
      main.getByRole('heading', { name: solverPage.finalAlgorithm }),
    ).toBeVisible();

    const numberedSections = main.locator('h2').filter({ hasText: /^\d+\./ });
    await expect(numberedSections).toHaveCount(12);
    expect(
      await main.locator('mjx-container[display="true"]').count(),
    ).toBeGreaterThanOrEqual(20);
    await expect(main).not.toContainText('$$');
  }
});
