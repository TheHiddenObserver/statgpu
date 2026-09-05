import { expect, test } from '@playwright/test';

const siteRoot = 'http://127.0.0.1:4173/statgpu/';

test.describe('Documentation navigation', () => {
  test('uses English by default and keeps Chinese as an optional locale', async ({
    page,
  }) => {
    await page.goto(siteRoot);

    const englishHero = page.locator('.VPHero');
    await expect(englishHero).toContainText('Statistical computing, accelerated');
    await expect(page.locator('.VPFeatures')).toContainText(
      'Learn the method, not just the API',
    );
    await expect(englishHero).not.toContainText('让统计计算更快');

    const lightImage = englishHero.locator('img.light');
    const darkImage = englishHero.locator('img.dark');
    await expect(lightImage).toHaveAttribute(
      'src',
      '/statgpu/images/statgpu-compute-hero-light.jpg',
    );
    await expect(darkImage).toHaveAttribute(
      'src',
      '/statgpu/images/statgpu-compute-hero.webp',
    );
    const lightWidth = await lightImage.evaluate(element => {
      if (!(element instanceof HTMLImageElement)) {
        throw new Error('Expected the light hero asset to be an image');
      }
      return element.naturalWidth;
    });
    const darkWidth = await darkImage.evaluate(element => {
      if (!(element instanceof HTMLImageElement)) {
        throw new Error('Expected the dark hero asset to be an image');
      }
      return element.naturalWidth;
    });
    expect(lightWidth).toBe(768);
    expect(darkWidth).toBe(768);

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

    await page.getByRole('button', { name: 'Change language' }).click();
    await page
      .locator('.VPNavBarTranslations')
      .getByRole('link', { name: 'English' })
      .click();
    await expect(page).toHaveURL(siteRoot);
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

  test('keeps the model catalog focused and places specialized solvers on model pages', async ({
    page,
  }) => {
    await page.goto(siteRoot + 'en/models/');
    const modelCatalog = page.getByRole('main');
    await expect(modelCatalog).not.toContainText('Solver lookup');
    await modelCatalog.getByRole('link', { name: 'Ridge regression' }).click();
    await expect(page).toHaveURL(siteRoot + 'en/models/ridge');
    await expect(
      page.getByRole('heading', { name: 'Solver support' }),
    ).toBeVisible();

    await page.goto(siteRoot + 'en/models/quantile');
    const quantileMain = page.getByRole('main');
    await expect(
      quantileMain.getByRole('heading', {
        name: 'Quantile proximal IRLS (SCAD/MCP)',
      }),
    ).toBeVisible();
    await expect(
      quantileMain.getByRole('heading', {
        name: 'Quantile coordinate descent',
      }),
    ).toBeVisible();
    await expect(quantileMain).toContainText(
      'proximal_irls_quantile_solver',
    );
    expect(await quantileMain.locator('mjx-container').count()).toBeGreaterThan(
      10,
    );

    await page.goto(siteRoot + 'en/guides/solver-algorithms');
    const generalSolverMain = page.getByRole('main');
    await expect(generalSolverMain).not.toContainText(
      'Quantile coordinate descent',
    );
    await expect(generalSolverMain).not.toContainText(
      'proximal_irls_quantile_solver',
    );
    expect(
      await generalSolverMain.locator('mjx-container').count(),
    ).toBeGreaterThan(10);

    await page.goto(siteRoot + 'en/models/logistic-regression');
    const logisticMain = page.getByRole('main');
    await expect(logisticMain).toContainText(
      'does not expose a public solver parameter',
    );
    await expect(logisticMain).toContainText('fixed IRLS');

    await page.goto(siteRoot + 'cn/models/');
    await expect(page.getByRole('main')).not.toContainText('求解器速查');
    await page
      .getByRole('main')
      .getByRole('link', { name: '线性回归' })
      .click();
    await expect(page.locator('main h1')).toContainText(
      '线性回归（LinearRegression）',
    );

    await page.goto(siteRoot + 'cn/models/quantile');
    await expect(
      page.getByRole('heading', { name: '分位数近端 IRLS（SCAD/MCP）' }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: '分位数坐标下降' }),
    ).toBeVisible();
  });

  test('exposes solver documentation and GLM inference references', async ({
    page,
  }) => {
    await page.goto(siteRoot + 'en/models/generalized-linear-model');
    await expect(
      page.locator('.VPSidebar a[href="/statgpu/en/guides/solver-algorithms"]'),
    ).toHaveCount(1);
    await expect(
      page.locator('.VPSidebar a[href="/statgpu/en/guides/solver-penalty-matrix"]'),
    ).toHaveCount(1);
    await expect(page.getByRole('main')).toContainText(
      'PoissonRegression Results',
    );
    await page
      .getByRole('main')
      .getByRole('link', { name: 'GLM covariance and inference reference' })
      .click();
    await expect(page).toHaveURL(
      siteRoot + 'en/models/glm-family-reference#covariance-by-family-link-and-covariance-type',
    );
    await expect(
      page.getByRole('heading', {
        name: 'Covariance by family, link, and covariance type',
      }),
    ).toBeVisible();

    await page.goto(siteRoot + 'cn/guides/solver-algorithms');
    await expect(page.locator('main h1')).toContainText('求解器算法');
    await expect(
      page.locator('.VPSidebar a[href="/statgpu/cn/guides/solver-penalty-matrix"]'),
    ).toHaveCount(1);
  });

  test('publishes algorithm references and detailed GLM family setup', async ({
    page,
  }) => {
    await page.goto(siteRoot + 'en/guides/solver-algorithms');
    const solverMain = page.getByRole('main');
    await expect(solverMain).not.toContainText('was missing from the previous page');
    await expect(solverMain.locator('table').first()).not.toContainText(
      'Primary reference',
    );
    await expect(solverMain).not.toContainText('Primary reference');
    await expect(solverMain).toContainText(
      'following Beck and Teboulle (2009)',
    );
    await expect(solverMain).toContainText('follow Boyd et al. (2011)');
    await expect(solverMain).toContainText(
      'Proximal Newton-type methods for minimizing composite functions',
    );
    await expect(solverMain).toContainText(
      'Iteratively reweighted least squares for maximum likelihood estimation',
    );
    await expect(solverMain).toContainText(
      'Ridge regression: Biased estimation for nonorthogonal problems',
    );
    await expect(solverMain).not.toContainText('Quantile coordinate descent');
    await expect(solverMain.locator('h2').last()).toContainText('References');

    await page.goto(siteRoot + 'cn/guides/solver-algorithms');
    const chineseSolverMain = page.getByRole('main');
    await expect(chineseSolverMain).not.toContainText(
      '\u65e7\u9875\u9762\u9057\u6f0f',
    );
    await expect(
      chineseSolverMain.locator('table').first(),
    ).not.toContainText('\u4e3b\u8981\u53c2\u8003\u6587\u732e');
    await expect(chineseSolverMain).not.toContainText(
      '\u4e3b\u8981\u53c2\u8003\u6587\u732e',
    );
    await expect(chineseSolverMain).toContainText(
      'Beck \u4e0e Teboulle\uff082009\uff09',
    );
    await expect(chineseSolverMain.locator('h2').last()).toContainText(
      '\u53c2\u8003\u6587\u732e',
    );

    await page.goto(siteRoot + 'en/models/generalized-linear-model');
    await page
      .getByRole('main')
      .getByRole('link', { name: 'Gaussian setup' })
      .click();
    await expect(page).toHaveURL(
      siteRoot + 'en/models/glm-family-reference#gaussian',
    );
    await expect(page.locator('main h2#gaussian')).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByRole('main')).toContainText(
      'GeneralizedLinearModel(family="gaussian", solver="newton", C=0)',
    );

    await page.goto(siteRoot + 'en/models/generalized-linear-model');
    await page
      .getByRole('main')
      .getByRole('link', { name: 'Binomial setup' })
      .click();
    await expect(page).toHaveURL(
      siteRoot + 'en/models/glm-family-reference#binomial',
    );
    await expect(page.locator('main h2#binomial')).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByRole('main')).toContainText(
      'it does not expose a trials denominator',
    );

    await page.goto(siteRoot + 'cn/models/generalized-linear-model');
    await page
      .getByRole('main')
      .getByRole('link', { name: 'Gaussian \u8bbe\u7f6e' })
      .click();
    await expect(page.locator('main h2#gaussian')).toBeVisible({
      timeout: 20_000,
    });
  });

  test('prefetches and searches bilingual documentation', async ({ page }) => {
    const englishIndex = page.waitForResponse(
      response =>
        response.url().includes('@localSearchIndexroot') &&
        response.status() === 200,
    );
    await page.goto(siteRoot + 'en/getting-started/quickstart');
    await englishIndex;

    await page.getByRole('button', { name: 'Search' }).click();
    const englishInput = page.locator('#localsearch-input');
    await englishInput.fill('solver algorithms');
    const englishResult = page.locator(
      `.VPLocalSearchBox a.result[href^='/statgpu/en/guides/solver-algorithms']`,
    );
    await expect(englishResult.first()).toBeVisible();
    await expect(page.locator('.toggle-layout-button')).toHaveCount(0);
    await englishResult.first().click();
    await expect(page).toHaveURL(
      /\/statgpu\/en\/guides\/solver-algorithms/,
    );
    await expect(page.locator('main h1')).toContainText('Solver algorithms');

    const chineseIndex = page.waitForResponse(
      response =>
        response.url().includes('@localSearchIndexcn') &&
        response.status() === 200,
    );
    await page.goto(siteRoot + 'cn/getting-started/quickstart');
    await chineseIndex;

    await page
      .getByRole('button', { name: '\u641c\u7d22\u6587\u6863' })
      .click();
    const chineseInput = page.locator('#localsearch-input');
    await expect(chineseInput).toHaveAttribute('placeholder', '\u641c\u7d22');
    await chineseInput.fill('\u6c42\u89e3\u5668\u7b97\u6cd5');
    const chineseResult = page.locator(
      `.VPLocalSearchBox a.result[href^='/statgpu/cn/guides/solver-algorithms']`,
    );
    await expect(chineseResult.first()).toBeVisible();
    await expect(page.locator('.search-keyboard-shortcuts')).toContainText(
      '\u5bfc\u822a',
    );
    await chineseResult.first().click();
    await expect(page).toHaveURL(
      /\/statgpu\/cn\/guides\/solver-algorithms/,
    );
    await expect(page.locator('main h1')).toContainText(
      '\u6c42\u89e3\u5668\u7b97\u6cd5',
    );
  });
});
