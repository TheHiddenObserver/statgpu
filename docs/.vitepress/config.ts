import { defineConfig } from 'vitepress';

function normalizeBase(value: string | undefined): string {
  const candidate = value?.trim() || '/statgpu/';
  if (!candidate.startsWith('/') || !candidate.endsWith('/')) {
    throw new Error('STATGPU_SITE_BASE must start and end with "/"');
  }
  return candidate.replace(/\/{2,}/g, '/');
}

const base = normalizeBase(process.env.STATGPU_SITE_BASE);

const englishSidebar = [
  {
    text: 'Getting started',
    items: [
      { text: 'Documentation home', link: '/en/' },
      { text: 'Quickstart', link: '/en/getting-started/quickstart' },
      { text: 'Usage', link: '/en/usage' },
      { text: 'Implemented methods', link: '/en/guides/implemented-methods' },
    ],
  },
  {
    text: 'Guides',
    items: [
      { text: 'Device and memory', link: '/en/guides/device-and-memory' },
      { text: 'Cross-validation', link: '/en/guides/cross-validation' },
      { text: 'Inference API', link: '/en/guides/inference-api' },
      { text: 'Benchmarks', link: '/en/guides/benchmarks' },
    ],
  },
  {
    text: 'Model reference',
    items: [
      { text: 'Model catalog', link: '/en/models/' },
      { text: 'Unsupervised learning', link: '/en/unsupervised/' },
      { text: 'Panel models', link: '/en/panel/' },
    ],
  },
  {
    text: 'Project',
    items: [
      { text: 'Changelog', link: '/en/changelog' },
    ],
  },
];

const chineseSidebar = [
  {
    text: '\u5feb\u901f\u5f00\u59cb',
    items: [
      { text: '\u6587\u6863\u9996\u9875', link: '/cn/' },
      { text: '\u5feb\u901f\u4e0a\u624b', link: '/cn/getting-started/quickstart' },
      { text: '\u4f7f\u7528\u8bf4\u660e', link: '/cn/usage' },
      { text: '\u5df2\u5b9e\u73b0\u65b9\u6cd5', link: '/cn/guides/implemented-methods' },
    ],
  },
  {
    text: '\u6307\u5357',
    items: [
      { text: '\u8bbe\u5907\u4e0e\u663e\u5b58', link: '/cn/guides/device-and-memory' },
      { text: '\u4ea4\u53c9\u9a8c\u8bc1', link: '/cn/guides/cross-validation' },
      { text: '\u63a8\u65ad API', link: '/cn/guides/inference-api' },
      { text: '\u6027\u80fd\u57fa\u51c6', link: '/cn/guides/benchmarks' },
    ],
  },
  {
    text: '\u6a21\u578b\u53c2\u8003',
    items: [
      { text: '\u6a21\u578b\u76ee\u5f55', link: '/cn/models/' },
      { text: '\u65e0\u76d1\u7763\u5b66\u4e60', link: '/cn/unsupervised/' },
      { text: '\u9762\u677f\u6a21\u578b', link: '/cn/panel/' },
    ],
  },
  {
    text: '\u9879\u76ee',
    items: [
      { text: '\u66f4\u65b0\u65e5\u5fd7', link: '/cn/changelog' },
    ],
  },
];

export default defineConfig({
  base,
  outDir: '../.site-dist',
  lang: 'en-US',
  title: 'statgpu',
  description: 'GPU-accelerated statistical methods with an sklearn-style API.',
  lastUpdated: true,
  cleanUrls: true,
  markdown: {
    math: true,
    config(md) {
      const defaultLinkOpen: NonNullable<
        typeof md.renderer.rules.link_open
      > =
        md.renderer.rules.link_open ??
        ((tokens, index, options, _env, self) =>
          self.renderToken(tokens, index, options));

      md.renderer.rules.link_open = (tokens, index, options, env, self) => {
        const href = tokens[index].attrGet('href');
        if (href === '/dashboard/' || href === '/dashboard') {
          // The dashboard is a separate Vite app. Bypass VitePress SPA routing
          // so navigation loads its assembled index.html instead of the docs 404.
          tokens[index].attrSet('href', base + 'dashboard/');
          tokens[index].attrSet('target', '_self');
        }
        return defaultLinkOpen(tokens, index, options, env, self);
      };
    },
  },
  // The dashboard is assembled after VitePress completes, then checked by
  // scripts/verify-site.mjs against the final deployment artifact.
  ignoreDeadLinks: [/^\/dashboard(?:\/|$)/],
  sitemap: {
    hostname:
      process.env.STATGPU_SITE_URL ||
      'https://thehiddenobserver.github.io/statgpu/',
  },
  transformPageData(pageData) {
    if (pageData.relativePath.startsWith('cn/')) {
      pageData.frontmatter.lang = 'zh-CN';
    }
  },
  transformHtml(code, id) {
    const normalizedId = id.replace(/\\/g, '/');
    if (!normalizedId.includes('/cn/')) return code;
    return code.replace(
      /<html lang="[^"]+"/,
      '<html lang="zh-CN"',
    );
  },
  head: [
    ['meta', { name: 'theme-color', content: '#3156a8' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'statgpu' }],
    [
      'meta',
      {
        property: 'og:description',
        content: 'GPU-accelerated statistical methods and reproducible benchmarks.',
      },
    ],
  ],
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      {
        text: 'Documentation',
        items: [
          { text: 'English', link: '/en/' },
          { text: '\u7b80\u4f53\u4e2d\u6587', link: '/cn/' },
        ],
      },
      { text: 'Dashboard', link: '/dashboard/', target: '_self' },
      { text: 'Changelog', link: '/en/changelog' },
    ],
    sidebar: {
      '/en/': englishSidebar,
      '/cn/': chineseSidebar,
    },
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/TheHiddenObserver/statgpu' },
    ],
    footer: {
      message: 'Released under the Apache-2.0 License.',
      copyright: 'Copyright (c) statgpu contributors',
    },
  },
});
