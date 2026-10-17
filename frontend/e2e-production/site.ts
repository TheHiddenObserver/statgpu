/** Match the site's configurable project-Pages or custom-domain base. */
const candidate = process.env.STATGPU_SITE_BASE?.trim() || '/statgpu/';
if (!candidate.startsWith('/') || !candidate.endsWith('/')) {
  throw new Error('STATGPU_SITE_BASE must start and end with "/"');
}

export const siteBase = candidate.replace(/\/{2,}/g, '/');
export const siteRoot = `http://127.0.0.1:4173${siteBase}`;
export const dashboardURL = `${siteRoot}dashboard/`;
