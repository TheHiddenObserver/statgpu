import { expect, test } from '@playwright/test';
import { siteBase, siteRoot } from './site';

for (const language of ['en', 'cn']) {
  test(`renders inline and display LaTeX on current ${language} model pages`, async ({ page }) => {
    for (const route of ['panel/random-effects', 'models/regression/linear-regression', 'models/regression/generalized-linear-model']) {
      await page.goto(`${siteRoot}${language}/${route}`);
      const main = page.getByRole('main');
      await expect(main.locator('mjx-container').first()).toBeVisible({ timeout: 60_000 });
      expect(await main.locator('mjx-container').count()).toBeGreaterThan(5);
      expect(await main.locator('mjx-container[display="true"]').count()).toBeGreaterThan(0);
      expect(await main.locator('mjx-container:not([display="true"])').count()).toBeGreaterThan(0);
      await expect(main).not.toContainText('$$');
    }
  });

  test(`renders and links the ${language} Lasso inference companion`, async ({ page }) => {
    await page.goto(`${siteRoot}${language}/models/regression/lasso`);
    await page.locator('main a[href$="lasso-inference"]').click();
    await expect(page).toHaveURL(`${siteRoot}${language}/models/regression/lasso-inference`);
    const main = page.getByRole('main');
    await expect(main.locator('h1')).toContainText('Lasso');
    await expect(main.locator('mjx-container').first()).toBeVisible({ timeout: 60_000 });
    await expect(main.locator('mjx-container[display="true"]')).toHaveCount(21);
    await expect(main.locator('[data-mml-node="merror"]')).toHaveCount(0);
    await expect(main).not.toContainText('$$');
    await expect(main).toContainText('simultaneous_target_mask');
    await expect(main).toContainText('fit_intercept=False');
    await expect(page.locator(`.VPSidebar a[href="${siteBase}${language}/models/regression/lasso-inference"]`)).toHaveCount(1);

    const peer = language === 'en' ? 'cn' : 'en';
    await main.locator(`a[href$="${peer}/models/regression/lasso-inference"]`).click();
    await expect(page).toHaveURL(`${siteRoot}${peer}/models/regression/lasso-inference`);
    await expect(page.locator('main mjx-container[display="true"]')).toHaveCount(21);
    await page.reload();
    await expect(page.locator('main h1')).toContainText('Lasso');
    await page.goBack();
    await expect(page).toHaveURL(`${siteRoot}${language}/models/regression/lasso-inference`);
    await expect(page.locator('main mjx-container[display="true"]')).toHaveCount(21);
  });

  test(`renders the current ${language} solver algorithms and preserved anchors`, async ({ page }) => {
    await page.goto(`${siteRoot}${language}/guides/solver-algorithms`);
    const main = page.getByRole('main');
    await expect(main.getByRole('heading', { name: '1. Proximal IRLS-CD' })).toBeVisible();
    await expect(main.locator('h2').filter({ hasText: /^10\. exact/ })).toBeVisible();
    await expect(main.locator('#quantile-low-level-solvers')).toHaveCount(1);
    expect(await main.locator('mjx-container[display="true"]').count()).toBeGreaterThanOrEqual(20);
    await expect(main).not.toContainText('$$');
  });
}
