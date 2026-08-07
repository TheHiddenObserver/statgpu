import { test, expect } from '@playwright/test';

test.describe('Audited source inventory', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  test('uses literal inventory-v2 labels', async ({ page }) => {
    const inventory = page.locator('.inventory-meta');
    await expect(inventory).toBeVisible();
    await expect(inventory).toContainText(/\d+ registered/);
    await expect(inventory).toContainText(/\d+ eligible/);
    await expect(inventory).toContainText(/\d+ non-ready/);
    await expect(inventory).toContainText(/\d+ historical\/excluded/);
    await expect(inventory).toHaveAttribute('title', /discovered JSON artifacts/);
    await expect(inventory).toHaveAttribute('title', /unclassified/);
  });

  test('links the inventory, catalog policy, and coverage matrix', async ({ page }) => {
    await expect(page.getByRole('link', { name: 'Source inventory (JSON)' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Catalog policy' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Coverage matrix' })).toBeVisible();
    await expect(page.locator('.dashboard-footer')).toContainText('Inventory 2.0');
  });
});
