import { expect, test } from '@playwright/test';
import { dashboardURL, siteBase, siteRoot } from './site';

test.describe('Documentation navigation', () => {
  test('uses English by default and offers Chinese home navigation', async ({ page }) => {
    await page.goto(siteRoot);
    await expect(page.locator('.VPHero')).toContainText('Statistical computing, accelerated');
    await page.getByRole('button', { name: 'Change language' }).click();
    await page.locator('.VPNavBarTranslations').getByRole('link', { name: '简体中文' }).click();
    await expect(page).toHaveURL(`${siteRoot}cn/`);
    await expect(page.locator('.VPHero')).toContainText('让统计计算更快');
    await page.getByRole('button', { name: 'Change language' }).click();
    await page.locator('.VPNavBarTranslations').getByRole('link', { name: 'English' }).click();
    await expect(page).toHaveURL(siteRoot);
  });

  test('opens the assembled dashboard and returns home after refresh', async ({ page }) => {
    await page.goto(siteRoot);
    const dashboardLinks = page.locator(`a[href="${siteBase}dashboard/"]`);
    expect(await dashboardLinks.count()).toBeGreaterThanOrEqual(2);
    for (const link of await dashboardLinks.all()) {
      await expect(link).toHaveAttribute('target', '_self');
    }
    await dashboardLinks.first().click();
    await expect(page).toHaveURL(dashboardURL);
    await expect(page.locator('.header:not(.header-loading)')).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText('PAGE NOT FOUND', { exact: true })).toHaveCount(0);
    await page.reload();
    await page.getByRole('link', { name: 'Back to statgpu home' }).click();
    await expect(page).toHaveURL(siteRoot);
    await expect(page.locator('.VPHero')).toBeVisible();
  });

  for (const language of ['en', 'cn']) {
    test(`${language} catalog links to canonical family landing pages and tutorials`, async ({ page }) => {
      await page.goto(`${siteRoot}${language}/models/README`);
      const main = page.getByRole('main');
      for (const family of ['regression', 'testing', 'covariance', 'nonparametric', 'feature-selection']) {
        await expect(main.locator(`a[href$="${family}/README"]`)).toHaveCount(1);
      }
      await main.locator('a[href$="regression/README"]').click();
      await expect(page).toHaveURL(`${siteRoot}${language}/models/regression/README`);
      await page.getByRole('main').getByRole('link', { name: 'SCAD', exact: true }).click();
      await expect(page).toHaveURL(`${siteRoot}${language}/models/regression/scad`);
      await expect(page.locator('main h1')).toContainText('SCAD');
      await expect(page.locator(`.VPSidebar a[href="${siteBase}${language}/models/regression/scad"]`)).toHaveCount(1);

      // The complete interface lives in reference/, while family pages teach
      // method choice. Neither link should revive the removed flat model URLs.
      await page.goto(`${siteRoot}${language}/models/regression/ridge`);
      await page.locator('main a[href$="reference/linear-model-api#ridge"]').click();
      await expect(page).toHaveURL(`${siteRoot}${language}/reference/linear-model-api#ridge`);
      await expect(page.locator('main h2#ridge')).toBeVisible();
      await page.reload();
      await expect(page.locator('main h2#ridge')).toBeVisible();
    });

    test(`${language} panel and unsupervised overviews retain README routes`, async ({ page }) => {
      await page.goto(`${siteRoot}${language}/panel/README`);
      await page.locator('main a[href$="pooled-ols"]').first().click();
      await expect(page).toHaveURL(`${siteRoot}${language}/panel/pooled-ols`);
      await expect(page.locator('main h1')).toContainText('PooledOLS');

      await page.goto(`${siteRoot}${language}/unsupervised/README`);
      await page.locator('main a[href$="agglomerative-clustering"]').first().click();
      await expect(page).toHaveURL(`${siteRoot}${language}/unsupervised/agglomerative-clustering`);
      await expect(page.locator('main h1')).toContainText('AgglomerativeClustering');
    });

    test(`${language} quantile guide links to the current low-level solver reference`, async ({ page }) => {
      await page.goto(`${siteRoot}${language}/models/regression/quantile`);
      await page.locator('main a[href$="solver-algorithms#quantile-low-level-solvers"]').first().click();
      await expect(page).toHaveURL(`${siteRoot}${language}/guides/solver-algorithms#quantile-low-level-solvers`);
      await expect(page.locator('main #quantile-low-level-solvers')).toHaveCount(1);
      await expect(page.getByRole('heading', { name: '1. Proximal IRLS-CD' })).toBeVisible();
      await expect(page.locator(`.VPSidebar a[href="${siteBase}${language}/guides/solver-penalty-matrix"]`)).toHaveCount(1);
    });
  }

  for (const forceFallback of [false, true]) {
    test(`prefetches and searches canonical documentation in both languages (${forceFallback ? 'gzip fallback' : 'native gzip'})`, async ({ page }) => {
      if (forceFallback) {
        await page.addInitScript(() => {
          Object.defineProperty(globalThis, 'DecompressionStream', {
            value: undefined, configurable: true, writable: true,
          });
        });
      }
    for (const locale of [
      { language: 'en', index: 'root', button: 'Search', query: 'solver algorithms', reference: 'Linear-model API reference' },
      { language: 'cn', index: 'cn', button: '搜索文档', query: '求解器算法', reference: '线性模型 API 参考' },
    ]) {
      // Start listening before navigation: the locale module must preload
      // without a click or hover on the search button.
      const prefetchedIndex = page.waitForResponse(response =>
        response.url().includes(`@localSearchIndex${locale.index}`) && response.status() === 200,
      );
      await page.goto(`${siteRoot}${locale.language}/getting-started/quickstart`);
      await prefetchedIndex;
      await page.getByRole('button', { name: locale.button }).click();
      const input = page.locator('#localsearch-input');
      if (locale.language === 'cn') {
        await expect(input).toHaveAttribute('placeholder', '搜索');
        await expect(page.locator('.search-keyboard-shortcuts')).toContainText('导航');
      }
      await input.fill(locale.query);
      const result = page.locator(
        `.VPLocalSearchBox a.result[href^="${siteBase}${locale.language}/guides/solver-algorithms"]`,
      ).first();
      await expect(result).toBeVisible();
      await expect(page.locator('.toggle-layout-button')).toHaveCount(0);
      await result.click();
      await expect(page).toHaveURL(url =>
        url.pathname.startsWith(`${siteBase}${locale.language}/guides/solver-algorithms`),
      );
      await expect(page.locator('main h1')).toContainText(
        locale.language === 'en' ? 'Solver Algorithms' : '求解器算法',
      );

      // A complete reference page must stay searchable alongside tutorials.
      await page.getByRole('button', { name: locale.button }).click();
      await page.locator('#localsearch-input').fill(locale.reference);
      const reference = page.locator(
        `.VPLocalSearchBox a.result[href^="${siteBase}${locale.language}/reference/linear-model-api"]`,
      ).first();
      await expect(reference).toBeVisible();
      await expect(page.locator('.toggle-layout-button')).toHaveCount(0);
      await reference.click();
      await expect(page).toHaveURL(url =>
        url.pathname === `${siteBase}${locale.language}/reference/linear-model-api`,
      );
      await expect(page.locator('main h1')).toContainText(locale.reference);
    }
    });
  }
});
