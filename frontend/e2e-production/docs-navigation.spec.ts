import { expect, test } from '@playwright/test';

const siteRoot = 'http://127.0.0.1:4173/statgpu/';

test.describe('Documentation navigation', () => {
  test('presents a bilingual home page with a loaded technical illustration', async ({
    page,
  }) => {
    await page.goto(siteRoot);

    const hero = page.locator('.VPHero');
    await expect(hero).toContainText('Statistical computing, accelerated');
    await expect(hero).toContainText('GPU 加速统计方法');
    await expect(hero).toContainText('English Docs / 英文文档');
    await expect(hero).toContainText('中文文档 / Chinese Docs');

    const image = hero.locator('img');
    await expect(image).toHaveAttribute(
      'src',
      '/statgpu/images/statgpu-compute-hero.webp',
    );
    expect(await image.evaluate(element => element.naturalWidth)).toBe(768);
  });

  test('opens the assembled dashboard instead of the VitePress 404 page', async ({
    page,
  }) => {
    await page.goto(siteRoot);

    const dashboardLinks = page.locator(
      'a[href="/statgpu/dashboard/"]',
    );
    const dashboardLinkCount = await dashboardLinks.count();
    expect(dashboardLinkCount).toBeGreaterThanOrEqual(2);
    for (let index = 0; index < dashboardLinkCount; index += 1) {
      await expect(dashboardLinks.nth(index)).toHaveAttribute('target', '_self');
    }

    await dashboardLinks.first().click();
    await expect(page).toHaveURL(siteRoot + 'dashboard/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText('PAGE NOT FOUND', { exact: true })).toHaveCount(0);
  });

  test('links model catalogs directly to individual reference pages', async ({
    page,
  }) => {
    await page.goto(siteRoot + 'en/models/');
    const modelLinks = page.locator(
      'main a[href^="./"]',
    );
    expect(await modelLinks.count()).toBeGreaterThanOrEqual(29);
    await page.getByRole('main').getByRole('link', { name: 'SCAD' }).click();
    await expect(page).toHaveURL(siteRoot + 'en/models/scad');
    await expect(page.locator('main h1')).toContainText('SCAD');
    await expect(
      page.locator('.VPSidebar a[href="/statgpu/en/models/scad"]'),
    ).toHaveCount(1);
    await expect(
      page.locator('.VPSidebar a[href="/statgpu/en/unsupervised/pca"]'),
    ).toHaveCount(1);
    await expect(
      page.locator('.VPSidebar a[href="/statgpu/en/panel/panel-ols"]'),
    ).toHaveCount(1);

    await page.goto(siteRoot + 'en/unsupervised/');
    await page
      .getByRole('main')
      .getByRole('link', { name: 'Agglomerative clustering' })
      .click();
    await expect(page).toHaveURL(
      siteRoot + 'en/unsupervised/agglomerative-clustering',
    );
    await expect(page.locator('main h1')).toContainText(
      'AgglomerativeClustering',
    );

    await page.goto(siteRoot + 'en/panel/');
    await page.getByRole('main').getByRole('link', { name: 'Pooled OLS' }).click();
    await expect(page).toHaveURL(siteRoot + 'en/panel/pooled-ols');
    await expect(page.locator('main h1')).toContainText('PooledOLS');

    await page.goto(siteRoot + 'cn/models/');
    expect(
      await page.locator('main a[href^="./"]').count(),
    ).toBeGreaterThanOrEqual(29);
    const ridgeLink = page.locator('main a[href="./ridge"]');
    await expect(ridgeLink).toHaveText('岭回归');
    await ridgeLink.click();
    await expect(page).toHaveURL(siteRoot + 'cn/models/ridge');
    await expect(page.locator('main h1')).toContainText('Ridge');
  });
});
