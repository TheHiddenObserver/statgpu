import { defineConfig } from 'vitepress';

function normalizeBase(value: string | undefined): string {
  const candidate = value?.trim() || '/statgpu/';
  if (!candidate.startsWith('/') || !candidate.endsWith('/')) {
    throw new Error('STATGPU_SITE_BASE must start and end with "/"');
  }
  return candidate.replace(/\/{2,}/g, '/');
}

const base = normalizeBase(process.env.STATGPU_SITE_BASE);

function englishModelReference() {
  return {
    text: 'Model reference',
    collapsed: false,
    items: [
      { text: 'Model directory', link: '/en/models/' },
      { text: 'Current model overview', link: '/en/models/README' },
      {
        text: 'Regression and GLM',
        collapsed: true,
        items: [
          { text: 'Linear regression', link: '/en/models/linear-regression' },
          { text: 'Generalized linear models', link: '/en/models/generalized-linear-model' },
          { text: 'Logistic regression', link: '/en/models/logistic-regression' },
          { text: 'Poisson regression', link: '/en/models/poisson-regression' },
          { text: 'Ordered models', link: '/en/models/ordered' },
          { text: 'Quantile regression', link: '/en/models/quantile' },
          { text: 'Robust regression', link: '/en/models/robust' },
          { text: 'Cox proportional hazards', link: '/en/models/coxph' },
        ],
      },
      {
        text: 'Regularization',
        collapsed: true,
        items: [
          { text: 'Feature selection', link: '/en/models/feature-selection' },
          { text: 'Ridge', link: '/en/models/ridge' },
          { text: 'Lasso', link: '/en/models/lasso' },
          { text: 'Elastic Net', link: '/en/models/elastic-net' },
          { text: 'Adaptive Lasso', link: '/en/models/adaptive-lasso' },
          { text: 'SCAD', link: '/en/models/scad' },
          { text: 'MCP', link: '/en/models/mcp' },
          { text: 'Knockoff filters', link: '/en/models/knockoff' },
        ],
      },
      {
        text: 'Statistical modules',
        collapsed: true,
        items: [
          { text: 'ANOVA', link: '/en/models/anova' },
          { text: 'Covariance', link: '/en/models/covariance' },
          { text: 'Nonparametric methods', link: '/en/models/nonparametric' },
          { text: 'Kernel methods', link: '/en/models/kernel-methods' },
          { text: 'Splines', link: '/en/models/splines' },
          { text: 'GAM / semiparametric', link: '/en/models/semiparametric' },
          { text: 'Multiple testing', link: '/en/models/multiple-testing' },
          { text: 'Loss functions', link: '/en/models/losses' },
        ],
      },
      {
        text: 'Unsupervised learning',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/unsupervised/' },
          { text: 'PCA', link: '/en/unsupervised/pca' },
          { text: 'Incremental PCA', link: '/en/unsupervised/incremental-pca' },
          { text: 'Truncated SVD', link: '/en/unsupervised/truncated-svd' },
          { text: 'NMF', link: '/en/unsupervised/nmf' },
          { text: 'MiniBatch NMF', link: '/en/unsupervised/minibatch-nmf' },
          { text: 'K-Means', link: '/en/unsupervised/kmeans' },
          { text: 'MiniBatch K-Means', link: '/en/unsupervised/minibatch-kmeans' },
          { text: 'Agglomerative clustering', link: '/en/unsupervised/agglomerative-clustering' },
          { text: 'DBSCAN', link: '/en/unsupervised/dbscan' },
          { text: 'Gaussian mixture', link: '/en/unsupervised/gaussian-mixture' },
          { text: 't-SNE', link: '/en/unsupervised/tsne' },
          { text: 'UMAP', link: '/en/unsupervised/umap' },
        ],
      },
      {
        text: 'Panel models',
        collapsed: true,
        items: [
          { text: 'Overview', link: '/en/panel/' },
          { text: 'Pooled OLS', link: '/en/panel/pooled-ols' },
          { text: 'Panel OLS', link: '/en/panel/panel-ols' },
          { text: 'Between OLS', link: '/en/panel/between-ols' },
          { text: 'Random effects', link: '/en/panel/random-effects' },
          { text: 'First-difference OLS', link: '/en/panel/first-difference-ols' },
          { text: 'Fama-MacBeth', link: '/en/panel/fama-macbeth' },
          { text: 'Covariance', link: '/en/panel/covariance' },
          { text: 'Fit statistics', link: '/en/panel/fit-statistics' },
          { text: 'Diagnostics', link: '/en/panel/diagnostics' },
        ],
      },
    ],
  };
}

function chineseModelReference() {
  return {
    text: '模型参考',
    collapsed: false,
    items: [
      { text: '模型目录', link: '/cn/models/' },
      { text: '当前模型总览', link: '/cn/models/README' },
      {
        text: '回归与 GLM',
        collapsed: true,
        items: [
          { text: '线性回归', link: '/cn/models/linear-regression' },
          { text: '广义线性模型', link: '/cn/models/generalized-linear-model' },
          { text: 'Logistic 回归', link: '/cn/models/logistic-regression' },
          { text: 'Poisson 回归', link: '/cn/models/poisson-regression' },
          { text: '有序模型', link: '/cn/models/ordered' },
          { text: '分位数回归', link: '/cn/models/quantile' },
          { text: '稳健回归', link: '/cn/models/robust' },
          { text: 'Cox 比例风险', link: '/cn/models/coxph' },
        ],
      },
      {
        text: '正则化',
        collapsed: true,
        items: [
          { text: '特征选择', link: '/cn/models/feature-selection' },
          { text: 'Ridge', link: '/cn/models/ridge' },
          { text: 'Lasso', link: '/cn/models/lasso' },
          { text: 'Elastic Net', link: '/cn/models/elastic-net' },
          { text: 'Adaptive Lasso', link: '/cn/models/adaptive-lasso' },
          { text: 'SCAD', link: '/cn/models/scad' },
          { text: 'MCP', link: '/cn/models/mcp' },
          { text: 'Knockoff 筛选', link: '/cn/models/knockoff' },
        ],
      },
      {
        text: '统计模块',
        collapsed: true,
        items: [
          { text: '方差分析', link: '/cn/models/anova' },
          { text: '协方差估计', link: '/cn/models/covariance' },
          { text: '非参数方法', link: '/cn/models/nonparametric' },
          { text: '核方法', link: '/cn/models/kernel-methods' },
          { text: '样条', link: '/cn/models/splines' },
          { text: 'GAM / 半参数', link: '/cn/models/semiparametric' },
          { text: '多重检验', link: '/cn/models/multiple-testing' },
          { text: '损失函数', link: '/cn/models/losses' },
        ],
      },
      {
        text: '无监督学习',
        collapsed: true,
        items: [
          { text: '总览', link: '/cn/unsupervised/' },
          { text: 'PCA', link: '/cn/unsupervised/pca' },
          { text: 'Incremental PCA', link: '/cn/unsupervised/incremental-pca' },
          { text: 'Truncated SVD', link: '/cn/unsupervised/truncated-svd' },
          { text: 'NMF', link: '/cn/unsupervised/nmf' },
          { text: 'MiniBatch NMF', link: '/cn/unsupervised/minibatch-nmf' },
          { text: 'K-Means', link: '/cn/unsupervised/kmeans' },
          { text: 'MiniBatch K-Means', link: '/cn/unsupervised/minibatch-kmeans' },
          { text: '层次聚类', link: '/cn/unsupervised/agglomerative-clustering' },
          { text: 'DBSCAN', link: '/cn/unsupervised/dbscan' },
          { text: '高斯混合', link: '/cn/unsupervised/gaussian-mixture' },
          { text: 't-SNE', link: '/cn/unsupervised/tsne' },
          { text: 'UMAP', link: '/cn/unsupervised/umap' },
        ],
      },
      {
        text: '面板模型',
        collapsed: true,
        items: [
          { text: '总览', link: '/cn/panel/' },
          { text: 'Pooled OLS', link: '/cn/panel/pooled-ols' },
          { text: 'Panel OLS', link: '/cn/panel/panel-ols' },
          { text: 'Between OLS', link: '/cn/panel/between-ols' },
          { text: '随机效应', link: '/cn/panel/random-effects' },
          { text: '一阶差分 OLS', link: '/cn/panel/first-difference-ols' },
          { text: 'Fama-MacBeth', link: '/cn/panel/fama-macbeth' },
          { text: '协方差估计', link: '/cn/panel/covariance' },
          { text: '拟合统计量', link: '/cn/panel/fit-statistics' },
          { text: '模型诊断', link: '/cn/panel/diagnostics' },
        ],
      },
    ],
  };
}

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
  englishModelReference(),
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
  chineseModelReference(),
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
  locales: {
    root: {
      label: 'Choose language',
      lang: 'en-US',
      link: '/',
      description: 'Choose the English or Chinese statgpu documentation.',
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      description: 'GPU-accelerated statistical methods with an sklearn-style API.',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          {
            text: 'Documentation',
            items: [
              { text: 'Quickstart', link: '/en/getting-started/quickstart' },
              { text: 'Model catalog', link: '/en/models/' },
              { text: 'Implemented methods', link: '/en/guides/implemented-methods' },
            ],
          },
          { text: 'Dashboard', link: '/dashboard/', target: '_self' },
          { text: 'Changelog', link: '/en/changelog' },
        ],
        sidebar: { '/en/': englishSidebar },
        footer: {
          message: 'Released under the Apache-2.0 License',
          copyright: 'Copyright (c) statgpu contributors',
        },
      },
    },
    cn: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/cn/',
      description: '以 sklearn 风格 API 提供 GPU 加速统计方法。',
      themeConfig: {
        nav: [
          { text: '首页', link: '/cn/' },
          {
            text: '文档',
            items: [
              { text: '快速上手', link: '/cn/getting-started/quickstart' },
              { text: '模型目录', link: '/cn/models/' },
              { text: '已实现方法', link: '/cn/guides/implemented-methods' },
            ],
          },
          { text: '基准面板', link: '/dashboard/', target: '_self' },
          { text: '更新日志', link: '/cn/changelog' },
        ],
        sidebar: { '/cn/': chineseSidebar },
        footer: {
          message: '采用 Apache-2.0 许可证发布',
          copyright: 'Copyright (c) statgpu 贡献者',
        },
      },
    },
  },
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
    // Not every historical document has a translated counterpart. Keep the
    // global language menu on stable locale home pages; paired model guides
    // provide direct language links in their page headers.
    i18nRouting: false,
    nav: [
      { text: 'Choose language', link: '/' },
      {
        text: 'Documentation language',
        items: [
          { text: 'English', link: '/en/' },
          { text: '\u7b80\u4f53\u4e2d\u6587', link: '/cn/' },
        ],
      },
    ],
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/TheHiddenObserver/statgpu' },
    ],
    footer: {
      message: 'Select English or 简体中文 to continue',
      copyright: 'Copyright (c) statgpu contributors',
    },
  },
});
