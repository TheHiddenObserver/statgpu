import { defineConfig } from 'vitepress';
import path from 'node:path';
import { compressedLocalSearchPlugin } from '../../scripts/search-index-compression.mjs';
import { installHeadingAnchors } from './site-heading-anchors.mjs';
import { installSourceLinks } from './site-source-links.mjs';

const docsRoot = path.resolve(process.cwd(), 'docs');

function normalizeBase(value: string | undefined): string {
  const candidate = value?.trim() || '/statgpu/';
  if (!candidate.startsWith('/') || !candidate.endsWith('/')) {
    throw new Error('STATGPU_SITE_BASE must start and end with "/"');
  }
  return candidate.replace(/\/{2,}/g, '/');
}

const base = normalizeBase(process.env.STATGPU_SITE_BASE);

// Keep local search focused on public user documentation. These pages remain
// published and directly reachable, but indexing them adds large maintenance
// records and duplicate benchmark listings that dilute user-facing results.
const excludedSearchPages = [
  /^benchmark-dashboard\//,
  /^(?:en|cn)\/benchmarks\.md$/,
  /^(?:en|cn)\/changelog(?:-history-through-[^/]+)?\.md$/,
  /^en\/guides\/statgpu_benchmark_dashboard_next_phase_plan\.md$/,
  /^website-deployment\.md$/,
];

function isPublicSearchPage(relativePath: string): boolean {
  return !excludedSearchPages.some(pattern => pattern.test(relativePath));
}

function englishModelReference() {
  return {
  "text": "Model reference",
  "collapsed": false,
  "items": [
    {
      "text": "Model directory",
      "link": "/en/models/README"
    },
    {
      "text": "Regression",
      "collapsed": true,
      "items": [
        {
          "text": "Regression",
          "link": "/en/models/regression/README"
        },
        {
          "text": "Adaptive Lasso",
          "link": "/en/models/regression/adaptive-lasso"
        },
        {
          "text": "Elastic Net",
          "link": "/en/models/regression/elastic-net"
        },
        {
          "text": "GeneralizedLinearModel and Penalized GLM",
          "link": "/en/models/regression/generalized-linear-model"
        },
        {
          "text": "Lasso",
          "link": "/en/models/regression/lasso"
        },
        {
          "text": "Lasso inference",
          "link": "/en/models/regression/lasso-inference"
        },
        {
          "text": "LinearRegression",
          "link": "/en/models/regression/linear-regression"
        },
        {
          "text": "LogisticRegression",
          "link": "/en/models/regression/logistic-regression"
        },
        {
          "text": "MCP",
          "link": "/en/models/regression/mcp"
        },
        {
          "text": "Ordered Generalized Linear Models (Logit/Probit)",
          "link": "/en/models/regression/ordered"
        },
        {
          "text": "PoissonRegression",
          "link": "/en/models/regression/poisson-regression"
        },
        {
          "text": "Quantile Regression",
          "link": "/en/models/regression/quantile"
        },
        {
          "text": "Ridge",
          "link": "/en/models/regression/ridge"
        },
        {
          "text": "Robust Regression",
          "link": "/en/models/regression/robust"
        },
        {
          "text": "SCAD",
          "link": "/en/models/regression/scad"
        }
      ]
    },
    {
      "text": "Statistical testing",
      "collapsed": true,
      "items": [
        {
          "text": "Statistical Testing",
          "link": "/en/models/testing/README"
        },
        {
          "text": "Classical one-way ANOVA",
          "link": "/en/models/testing/anova-oneway"
        },
        {
          "text": "Post-hoc comparisons: Tukey and Bonferroni Welch tests",
          "link": "/en/models/testing/anova-posthoc"
        },
        {
          "text": "Balanced two-way ANOVA",
          "link": "/en/models/testing/anova-twoway"
        },
        {
          "text": "Welch one-way ANOVA",
          "link": "/en/models/testing/anova-welch"
        },
        {
          "text": "ANOVA: comparing group means",
          "link": "/en/models/testing/anova"
        },
        {
          "text": "False discovery rate control: BH and BY",
          "link": "/en/models/testing/multiple-testing-fdr"
        },
        {
          "text": "Family-wise error control: Bonferroni, Holm and Hochberg",
          "link": "/en/models/testing/multiple-testing-fwer"
        },
        {
          "text": "Multiple testing and p-value combination",
          "link": "/en/models/testing/multiple-testing"
        },
        {
          "text": "Combine p-values: Fisher, Stouffer and Cauchy",
          "link": "/en/models/testing/pvalue-combination"
        }
      ]
    },
    {
      "text": "Covariance estimation",
      "collapsed": true,
      "items": [
        {
          "text": "Covariance Estimation",
          "link": "/en/models/covariance/README"
        },
        {
          "text": "Empirical Covariance: Learn the Unregularized Baseline",
          "link": "/en/models/covariance/covariance-empirical"
        },
        {
          "text": "GraphicalLassoCV: choose a sparse-model penalty with held-out data",
          "link": "/en/models/covariance/covariance-graphical-lasso-cv"
        },
        {
          "text": "GraphicalLasso: a sparse map of conditional relationships",
          "link": "/en/models/covariance/covariance-graphical-lasso"
        },
        {
          "text": "Ledoit–Wolf: Let the Data Choose Shrinkage",
          "link": "/en/models/covariance/covariance-ledoit-wolf"
        },
        {
          "text": "MinCovDet: estimate the center and spread despite outlying rows",
          "link": "/en/models/covariance/covariance-min-cov-det"
        },
        {
          "text": "OAS: Gaussian-Based Shrinkage for Small Samples",
          "link": "/en/models/covariance/covariance-oas"
        },
        {
          "text": "Fixed Shrinkage: Trade Some Detail for Stability",
          "link": "/en/models/covariance/covariance-shrunk"
        }
      ]
    },
    {
      "text": "Nonparametric methods",
      "collapsed": true,
      "items": [
        {
          "text": "Nonparametric Methods",
          "link": "/en/models/nonparametric/README"
        },
        {
          "text": "Kernel Methods",
          "link": "/en/models/nonparametric/kernel-methods"
        },
        {
          "text": "Nystroem Kernel Approximation",
          "link": "/en/models/nonparametric/kernel-nystroem"
        },
        {
          "text": "Kernel PCA",
          "link": "/en/models/nonparametric/kernel-pca"
        },
        {
          "text": "Kernel Ridge Cross-Validation",
          "link": "/en/models/nonparametric/kernel-ridge-cv"
        },
        {
          "text": "Kernel Ridge Regression",
          "link": "/en/models/nonparametric/kernel-ridge"
        },
        {
          "text": "Kernel density estimation and local regression",
          "link": "/en/models/nonparametric/nonparametric"
        },
        {
          "text": "GAM (Generalized Additive Model)",
          "link": "/en/models/nonparametric/semiparametric"
        },
        {
          "text": "Spline Basis Functions",
          "link": "/en/models/nonparametric/splines"
        }
      ]
    },
    {
      "text": "Feature selection",
      "collapsed": true,
      "items": [
        {
          "text": "Feature Selection",
          "link": "/en/models/feature-selection/README"
        },
        {
          "text": "Fixed-X Knockoff Selection",
          "link": "/en/models/feature-selection/knockoff-fixed-x"
        },
        {
          "text": "Model-X Knockoff Selection",
          "link": "/en/models/feature-selection/knockoff-model-x"
        },
        {
          "text": "Knockoff Feature Selection",
          "link": "/en/models/feature-selection/knockoff"
        }
      ]
    },
    {
      "text": "Unsupervised learning",
      "collapsed": true,
      "items": [
        {
          "text": "Unsupervised Learning",
          "link": "/en/unsupervised/README"
        },
        {
          "text": "AgglomerativeClustering",
          "link": "/en/unsupervised/agglomerative-clustering"
        },
        {
          "text": "Unsupervised estimator API reference",
          "link": "/en/unsupervised/api-reference"
        },
        {
          "text": "DBSCAN",
          "link": "/en/unsupervised/dbscan"
        },
        {
          "text": "GaussianMixture",
          "link": "/en/unsupervised/gaussian-mixture"
        },
        {
          "text": "IncrementalPCA",
          "link": "/en/unsupervised/incremental-pca"
        },
        {
          "text": "KMeans",
          "link": "/en/unsupervised/kmeans"
        },
        {
          "text": "MiniBatchKMeans",
          "link": "/en/unsupervised/minibatch-kmeans"
        },
        {
          "text": "MiniBatchNMF",
          "link": "/en/unsupervised/minibatch-nmf"
        },
        {
          "text": "NMF",
          "link": "/en/unsupervised/nmf"
        },
        {
          "text": "PCA",
          "link": "/en/unsupervised/pca"
        },
        {
          "text": "TruncatedSVD",
          "link": "/en/unsupervised/truncated-svd"
        },
        {
          "text": "TSNE",
          "link": "/en/unsupervised/tsne"
        },
        {
          "text": "UMAP",
          "link": "/en/unsupervised/umap"
        }
      ]
    },
    {
      "text": "Panel models",
      "collapsed": true,
      "items": [
        {
          "text": "Panel Models",
          "link": "/en/panel/README"
        },
        {
          "text": "Panel Architecture",
          "link": "/en/panel/architecture"
        },
        {
          "text": "BetweenOLS",
          "link": "/en/panel/between-ols"
        },
        {
          "text": "Panel Covariance Estimators",
          "link": "/en/panel/covariance"
        },
        {
          "text": "Panel Diagnostics",
          "link": "/en/panel/diagnostics"
        },
        {
          "text": "FamaMacBeth",
          "link": "/en/panel/fama-macbeth"
        },
        {
          "text": "FirstDifferenceOLS",
          "link": "/en/panel/first-difference-ols"
        },
        {
          "text": "Panel Fit Statistics",
          "link": "/en/panel/fit-statistics"
        },
        {
          "text": "PanelOLS",
          "link": "/en/panel/panel-ols"
        },
        {
          "text": "PooledOLS",
          "link": "/en/panel/pooled-ols"
        },
        {
          "text": "RandomEffects",
          "link": "/en/panel/random-effects"
        }
      ]
    },
    {
      "text": "Survival analysis",
      "collapsed": true,
      "items": [
        {
          "text": "Overview",
          "link": "/en/models/survival/README"
        },
        {
          "text": "Cox proportional hazards",
          "link": "/en/models/coxph"
        }
      ]
    },
    {
      "text": "Complete API references",
      "collapsed": true,
      "items": [
        {
          "text": "ANOVA API Reference",
          "link": "/en/reference/anova-api"
        },
        {
          "text": "Covariance API reference",
          "link": "/en/reference/covariance-api"
        },
        {
          "text": "CoxPH device and CV diagnostics",
          "link": "/en/reference/coxph-diagnostics"
        },
        {
          "text": "Shared estimator API",
          "link": "/en/reference/estimator-api"
        },
        {
          "text": "Feature-selection API reference",
          "link": "/en/reference/feature-selection-api"
        },
        {
          "text": "Kernel methods API",
          "link": "/en/reference/kernel-methods-api"
        },
        {
          "text": "Linear-model API reference",
          "link": "/en/reference/linear-model-api"
        },
        {
          "text": "Loss Functions (LossBase)",
          "link": "/en/reference/losses"
        },
        {
          "text": "Survival and smoothing API reference",
          "link": "/en/reference/survival-smoothing-api"
        }
      ]
    }
  ]
};
}

function chineseModelReference() {
  return {
  "text": "模型参考",
  "collapsed": false,
  "items": [
    {
      "text": "模型目录",
      "link": "/cn/models/README"
    },
    {
      "text": "回归",
      "collapsed": true,
      "items": [
        {
          "text": "回归",
          "link": "/cn/models/regression/README"
        },
        {
          "text": "Adaptive Lasso",
          "link": "/cn/models/regression/adaptive-lasso"
        },
        {
          "text": "Elastic Net 弹性网络",
          "link": "/cn/models/regression/elastic-net"
        },
        {
          "text": "广义线性模型与带惩罚 GLM",
          "link": "/cn/models/regression/generalized-linear-model"
        },
        {
          "text": "Lasso",
          "link": "/cn/models/regression/lasso"
        },
        {
          "text": "Lasso 推断原理",
          "link": "/cn/models/regression/lasso-inference"
        },
        {
          "text": "LinearRegression",
          "link": "/cn/models/regression/linear-regression"
        },
        {
          "text": "LogisticRegression",
          "link": "/cn/models/regression/logistic-regression"
        },
        {
          "text": "MCP",
          "link": "/cn/models/regression/mcp"
        },
        {
          "text": "有序广义线性模型 (Ordered Logit/Probit)",
          "link": "/cn/models/regression/ordered"
        },
        {
          "text": "PoissonRegression",
          "link": "/cn/models/regression/poisson-regression"
        },
        {
          "text": "分位数回归",
          "link": "/cn/models/regression/quantile"
        },
        {
          "text": "Ridge",
          "link": "/cn/models/regression/ridge"
        },
        {
          "text": "稳健回归",
          "link": "/cn/models/regression/robust"
        },
        {
          "text": "SCAD",
          "link": "/cn/models/regression/scad"
        }
      ]
    },
    {
      "text": "统计检验",
      "collapsed": true,
      "items": [
        {
          "text": "统计检验",
          "link": "/cn/models/testing/README"
        },
        {
          "text": "经典单因素方差分析",
          "link": "/cn/models/testing/anova-oneway"
        },
        {
          "text": "事后比较：Tukey 与 Bonferroni 两两 Welch 检验",
          "link": "/cn/models/testing/anova-posthoc"
        },
        {
          "text": "平衡双因素方差分析",
          "link": "/cn/models/testing/anova-twoway"
        },
        {
          "text": "Welch 单因素方差分析",
          "link": "/cn/models/testing/anova-welch"
        },
        {
          "text": "ANOVA：比较组均值",
          "link": "/cn/models/testing/anova"
        },
        {
          "text": "错误发现率控制：BH 与 BY",
          "link": "/cn/models/testing/multiple-testing-fdr"
        },
        {
          "text": "族错误率控制：Bonferroni、Holm 与 Hochberg",
          "link": "/cn/models/testing/multiple-testing-fwer"
        },
        {
          "text": "多重检验与 P 值合并",
          "link": "/cn/models/testing/multiple-testing"
        },
        {
          "text": "P 值合并：Fisher、Stouffer 与 Cauchy",
          "link": "/cn/models/testing/pvalue-combination"
        }
      ]
    },
    {
      "text": "协方差估计",
      "collapsed": true,
      "items": [
        {
          "text": "协方差估计",
          "link": "/cn/models/covariance/README"
        },
        {
          "text": "经验协方差：理解未经正则化的基线",
          "link": "/cn/models/covariance/covariance-empirical"
        },
        {
          "text": "GraphicalLassoCV：用留出数据选择稀疏模型的惩罚",
          "link": "/cn/models/covariance/covariance-graphical-lasso-cv"
        },
        {
          "text": "GraphicalLasso：用稀疏精度矩阵理解条件关联",
          "link": "/cn/models/covariance/covariance-graphical-lasso"
        },
        {
          "text": "Ledoit–Wolf：让数据决定收缩强度",
          "link": "/cn/models/covariance/covariance-ledoit-wolf"
        },
        {
          "text": "MinCovDet：减轻异常观测对中心和协方差的影响",
          "link": "/cn/models/covariance/covariance-min-cov-det"
        },
        {
          "text": "OAS：利用高斯假设进行小样本收缩",
          "link": "/cn/models/covariance/covariance-oas"
        },
        {
          "text": "固定收缩：用适度偏差换取稳定性",
          "link": "/cn/models/covariance/covariance-shrunk"
        }
      ]
    },
    {
      "text": "非参数方法",
      "collapsed": true,
      "items": [
        {
          "text": "非参数方法",
          "link": "/cn/models/nonparametric/README"
        },
        {
          "text": "核方法",
          "link": "/cn/models/nonparametric/kernel-methods"
        },
        {
          "text": "Nystroem 核近似",
          "link": "/cn/models/nonparametric/kernel-nystroem"
        },
        {
          "text": "核主成分分析",
          "link": "/cn/models/nonparametric/kernel-pca"
        },
        {
          "text": "核岭回归交叉验证",
          "link": "/cn/models/nonparametric/kernel-ridge-cv"
        },
        {
          "text": "核岭回归",
          "link": "/cn/models/nonparametric/kernel-ridge"
        },
        {
          "text": "核密度估计与局部核回归",
          "link": "/cn/models/nonparametric/nonparametric"
        },
        {
          "text": "GAM（广义可加模型）",
          "link": "/cn/models/nonparametric/semiparametric"
        },
        {
          "text": "样条基函数",
          "link": "/cn/models/nonparametric/splines"
        }
      ]
    },
    {
      "text": "特征选择",
      "collapsed": true,
      "items": [
        {
          "text": "特征选择",
          "link": "/cn/models/feature-selection/README"
        },
        {
          "text": "Fixed-X Knockoff 特征选择",
          "link": "/cn/models/feature-selection/knockoff-fixed-x"
        },
        {
          "text": "Model-X Knockoff 特征选择",
          "link": "/cn/models/feature-selection/knockoff-model-x"
        },
        {
          "text": "Knockoff 特征选择",
          "link": "/cn/models/feature-selection/knockoff"
        }
      ]
    },
    {
      "text": "无监督学习",
      "collapsed": true,
      "items": [
        {
          "text": "无监督学习",
          "link": "/cn/unsupervised/README"
        },
        {
          "text": "AgglomerativeClustering",
          "link": "/cn/unsupervised/agglomerative-clustering"
        },
        {
          "text": "无监督估计器 API 参考",
          "link": "/cn/unsupervised/api-reference"
        },
        {
          "text": "DBSCAN",
          "link": "/cn/unsupervised/dbscan"
        },
        {
          "text": "GaussianMixture",
          "link": "/cn/unsupervised/gaussian-mixture"
        },
        {
          "text": "IncrementalPCA",
          "link": "/cn/unsupervised/incremental-pca"
        },
        {
          "text": "KMeans",
          "link": "/cn/unsupervised/kmeans"
        },
        {
          "text": "MiniBatchKMeans",
          "link": "/cn/unsupervised/minibatch-kmeans"
        },
        {
          "text": "MiniBatchNMF",
          "link": "/cn/unsupervised/minibatch-nmf"
        },
        {
          "text": "NMF",
          "link": "/cn/unsupervised/nmf"
        },
        {
          "text": "PCA",
          "link": "/cn/unsupervised/pca"
        },
        {
          "text": "TruncatedSVD",
          "link": "/cn/unsupervised/truncated-svd"
        },
        {
          "text": "TSNE",
          "link": "/cn/unsupervised/tsne"
        },
        {
          "text": "UMAP",
          "link": "/cn/unsupervised/umap"
        }
      ]
    },
    {
      "text": "面板模型",
      "collapsed": true,
      "items": [
        {
          "text": "面板模型",
          "link": "/cn/panel/README"
        },
        {
          "text": "面板模型架构",
          "link": "/cn/panel/architecture"
        },
        {
          "text": "BetweenOLS",
          "link": "/cn/panel/between-ols"
        },
        {
          "text": "面板模型协方差估计",
          "link": "/cn/panel/covariance"
        },
        {
          "text": "面板模型诊断",
          "link": "/cn/panel/diagnostics"
        },
        {
          "text": "Fama–MacBeth",
          "link": "/cn/panel/fama-macbeth"
        },
        {
          "text": "FirstDifferenceOLS",
          "link": "/cn/panel/first-difference-ols"
        },
        {
          "text": "面板拟合统计量",
          "link": "/cn/panel/fit-statistics"
        },
        {
          "text": "PanelOLS",
          "link": "/cn/panel/panel-ols"
        },
        {
          "text": "PooledOLS",
          "link": "/cn/panel/pooled-ols"
        },
        {
          "text": "RandomEffects",
          "link": "/cn/panel/random-effects"
        }
      ]
    },
    {
      "text": "生存分析",
      "collapsed": true,
      "items": [
        {
          "text": "概览",
          "link": "/cn/models/survival/README"
        },
        {
          "text": "Cox 比例风险",
          "link": "/cn/models/coxph"
        }
      ]
    },
    {
      "text": "完整 API 参考",
      "collapsed": true,
      "items": [
        {
          "text": "ANOVA API 参考",
          "link": "/cn/reference/anova-api"
        },
        {
          "text": "协方差 API 参考",
          "link": "/cn/reference/covariance-api"
        },
        {
          "text": "CoxPH 设备与 CV 诊断",
          "link": "/cn/reference/coxph-diagnostics"
        },
        {
          "text": "估计器共享 API",
          "link": "/cn/reference/estimator-api"
        },
        {
          "text": "特征选择 API 参考",
          "link": "/cn/reference/feature-selection-api"
        },
        {
          "text": "核方法 API",
          "link": "/cn/reference/kernel-methods-api"
        },
        {
          "text": "线性模型 API 参考",
          "link": "/cn/reference/linear-model-api"
        },
        {
          "text": "损失函数（LossBase）",
          "link": "/cn/reference/losses"
        },
        {
          "text": "生存分析与平滑方法 API 参考",
          "link": "/cn/reference/survival-smoothing-api"
        }
      ]
    }
  ]
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
      { text: 'Solver algorithms', link: '/en/guides/solver-algorithms' },
      { text: 'Solver and penalty matrix', link: '/en/guides/solver-penalty-matrix' },
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
      { text: '\u6c42\u89e3\u5668\u7b97\u6cd5', link: '/cn/guides/solver-algorithms' },
      { text: '\u6c42\u89e3\u5668\u4e0e\u60e9\u7f5a\u77e9\u9635', link: '/cn/guides/solver-penalty-matrix' },
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
  vite: { plugins: [compressedLocalSearchPlugin()] },
  outDir: '../.site-dist',
  // VitePress 1.6.4 builds its MiniSearch index by concurrently adding pages
  // to a shared index. Serializing page work makes the emitted local-search
  // modules byte-for-byte reproducible, which the deployment hash gate checks.
  buildConcurrency: 1,
  lang: 'en-US',
  title: 'statgpu',
  description: 'GPU-accelerated statistical methods with an sklearn-style API.',
  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      link: '/',
      description: 'GPU-accelerated statistical methods with an sklearn-style API.',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          {
            text: 'Documentation',
            items: [
              { text: 'Quickstart', link: '/en/getting-started/quickstart' },
              { text: 'Model catalog', link: '/en/models/README' },
              { text: 'Solver guide', link: '/en/guides/solver-algorithms' },
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
              { text: '模型目录', link: '/cn/models/README' },
              { text: '求解器指南', link: '/cn/guides/solver-algorithms' },
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
      installHeadingAnchors(md);
      installSourceLinks(md, { docsRoot, base });
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
      { text: 'Home', link: '/' },
      {
        text: 'Documentation',
        items: [
          { text: 'Quickstart', link: '/en/getting-started/quickstart' },
          { text: 'Model catalog', link: '/en/models/README' },
          { text: 'Solver guide', link: '/en/guides/solver-algorithms' },
          { text: 'Implemented methods', link: '/en/guides/implemented-methods' },
        ],
      },
      { text: 'Dashboard', link: '/dashboard/', target: '_self' },
      { text: 'Changelog', link: '/en/changelog' },
    ],
    search: {
      provider: 'local',
      options: {
        // The compact result list avoids loading page modules merely to render
        // excerpts. Full pages still remain searchable by their body text.
        detailedView: false,
        locales: {
          cn: {
            translations: {
              button: {
                buttonText: '\u641c\u7d22',
                buttonAriaLabel: '\u641c\u7d22\u6587\u6863',
              },
              modal: {
                displayDetails: '\u663e\u793a\u8be6\u7ec6\u5217\u8868',
                resetButtonTitle: '\u6e05\u7a7a\u641c\u7d22',
                backButtonTitle: '\u5173\u95ed\u641c\u7d22',
                noResultsText: '\u672a\u627e\u5230\u7ed3\u679c',
                footer: {
                  selectText: '\u9009\u62e9',
                  selectKeyAriaLabel: '\u56de\u8f66',
                  navigateText: '\u5bfc\u822a',
                  navigateUpKeyAriaLabel: '\u4e0a\u7bad\u5934',
                  navigateDownKeyAriaLabel: '\u4e0b\u7bad\u5934',
                  closeText: '\u5173\u95ed',
                  closeKeyAriaLabel: '\u9000\u51fa',
                },
              },
            },
          },
        },
        _render(src, env, md) {
          if (!isPublicSearchPage(env.relativePath)) return '';
          const html = md.render(src, env);
          return env.frontmatter?.search === false ? '' : html;
        },
      },
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/TheHiddenObserver/statgpu' },
    ],
    footer: {
      message: 'Released under the Apache-2.0 License',
      copyright: 'Copyright (c) statgpu contributors',
    },
  },
});