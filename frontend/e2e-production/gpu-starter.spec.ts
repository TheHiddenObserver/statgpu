import { expect, test } from '@playwright/test';
import { siteRoot } from './site';

for (const locale of ['en', 'cn']) {
  test(`${locale} GPU starter guide is reachable and its diagrams load`, async ({ page }) => {
    await page.goto(`${siteRoot}${locale}/`);
    const guideLink = page.locator('.vp-doc').getByRole('link', {
      name: locale === 'en' ? 'GPU starter guide' : 'GPU 入门指南', exact: true,
    });
    await expect(guideLink).toBeVisible();
    await guideLink.click();
    await expect(page).toHaveURL(`${siteRoot}${locale}/guides/when-to-use-gpu`);
    const main = page.getByRole('main');
    for (const diagram of ['gpu-work', 'gpu-journey']) {
      const image = main.locator(`img[src$="${diagram}-${locale}.svg"]`);
      await expect(image).toBeVisible();
      await expect(image).toHaveAttribute('alt', /\S.{20,}/);
      await expect.poll(() => image.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0)).toBe(true);
    }
    await expect(main.locator('#when-cpu-is-the-better-start')).toHaveCount(1);
    await expect(main.locator('#measure-your-own-workload')).toHaveCount(1);
    await page.reload();
    await expect(main.locator('#try-it-with-an-explicit-device')).toHaveCount(1);
    const peer = locale === 'en' ? 'cn' : 'en';
    await main.locator(`a[href$="${peer}/guides/when-to-use-gpu"]`).click();
    await expect(page).toHaveURL(`${siteRoot}${peer}/guides/when-to-use-gpu`);
    await page.goBack();
    await expect(page).toHaveURL(`${siteRoot}${locale}/guides/when-to-use-gpu`);
  });

  for (const theme of ['light', 'dark']) {
    test(`${locale} homepage and guide fit narrow screens (${theme})`, async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await page.addInitScript((value) => localStorage.setItem('vitepress-theme-appearance', value), theme);
      for (const route of [`${locale}/`, `${locale}/guides/when-to-use-gpu`]) {
        await page.goto(`${siteRoot}${route}`);
        await expect(page.locator('#VPContent')).toBeVisible();
        if (theme === 'dark') await expect(page.locator('html')).toHaveClass(/\bdark\b/);
        else await expect(page.locator('html')).not.toHaveClass(/\bdark\b/);
        const dimensions = await page.evaluate(() => ({
          width: document.documentElement.clientWidth,
          content: document.documentElement.scrollWidth,
        }));
        expect(dimensions.content).toBeLessThanOrEqual(dimensions.width + 1);
      }
    });
  }
}
