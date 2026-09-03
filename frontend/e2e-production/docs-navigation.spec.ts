import { expect, test } from '@playwright/test';

const siteRoot = 'http://127.0.0.1:4173/statgpu/';

test.describe('Documentation navigation', () => {
  test('presents separate English and Chinese home pages', async ({
    page,
  }) => {
    await page.goto(siteRoot);

    const selectorHero = page.locator('.VPHero');
    await expect(selectorHero).toContainText('Choose your documentation language');
    await expect(selectorHero).toContainText('English');
    await expect(selectorHero).toContainText('简体中文');

    const image = selectorHero.locator('img');
    await expect(image).toHaveAttribute(
      'src',
      '/statgpu/images/statgpu-compute-hero.webp',
    );
    expect(await image.evaluate(element => element.naturalWidth)).toBe(768);

    await page.goto(siteRoot + 'en/');
    const englishHero = page.locator('.VPHero');
    await expect(englishHero).toContainText('Statistical computing, accelerated');
    await expect(page.locator('.VPFeatures')).toContainText(
      'Learn the method, not just the API',
    );
    await expect(englishHero).not.toContainText('让统计计算更快');

    await page.getByRole('button', { name: 'Change language' }).click();
    await page
      .locator('.VPNavBarTranslations')
      .getByRole('link', { name: '简体中文' })
      .click();
    await expect(page).toHaveURL(siteRoot + 'cn/');
    const chineseHero = page.locator('.VPHero');
    await expect(chineseHero).toContainText('让统计计算更快');
    await expect(page.locator('.VPFeatures')).toContainText(
      '不只介绍 API，也讲清方法',
    );
    await expect(chineseHero).not.toContainText('Statistical computing, accelerated');
  });

  test('uses the beginner-oriented guide structure in both languages', async ({
    page,
  }) => {
    const englishGuides = [
      ['en/models/linear-regression', 'What problem does it solve?', 'Minimal runnable example'],
      ['en/models/generalized-linear-model', 'What problem do GLMs solve?', 'How to read the result'],
      ['en/models/scad', 'When to use it', 'Common pitfalls'],
      ['en/unsupervised/dbscan', 'Prepare the data first', 'Key parameters and how to choose them'],
    ];

    for (const [route, firstHeading, secondHeading] of englishGuides) {
      await page.goto(siteRoot + route);
      const main = page.getByRole('main');
      await expect(main.getByRole('heading', { name: firstHeading })).toBeVisible();
      await expect(main.getByRole('heading', { name: secondHeading })).toBeVisible();
      await expect(main).not.toContainText('Language switch:');
    }

    const chineseGuides = [
      ['cn/models/linear-regression', '它解决什么问题？', '最小可运行示例'],
      ['cn/models/generalized-linear-model', 'GLM 解决什么问题？', '如何读取结果？'],
      ['cn/models/scad', '什么时候使用？', '常见误区'],
      ['cn/unsupervised/dbscan', '先准备数据', '关键参数怎么选？'],
    ];

    for (const [route, firstHeading, secondHeading] of chineseGuides) {
      await page.goto(siteRoot + route);
      const main = page.getByRole('main');
      await expect(main.getByRole('heading', { name: firstHeading })).toBeVisible();
      await expect(main.getByRole('heading', { name: secondHeading })).toBeVisible();
      await expect(main).not.toContainText('语言切换');
    }
  });

  test('opens the assembled dashboard instead of the VitePress 404 page', async ({
    page,
  }) => {
    await page.goto(siteRoot + 'en/');

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
