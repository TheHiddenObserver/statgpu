import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { parseFragment } from 'parse5';
import { createMarkdownRenderer } from 'vitepress';
import { installSourceLinks } from '../docs/.vitepress/site-source-links.mjs';
import { createSourceLinkChecker } from './check-source-document-links.mjs';

// VitePress caches its renderer globally. Use a fresh Node process for each
// base so both tests exercise the actual configured renderer, not that cache.
if (!process.env.STATGPU_LINK_TEST_CHILD) {
  for (const base of ['/statgpu/', '/']) {
    test(`configured source-link renderer at ${base}`, () => {
      const env = { ...process.env, STATGPU_SITE_BASE: base, STATGPU_LINK_TEST_CHILD: '1' };
      // Do not inherit the outer test worker's private IPC context. Otherwise
      // a nested --test process can exit successfully without running tests.
      delete env.NODE_TEST_CONTEXT;
      const result = spawnSync(process.execPath, ['--test', '--test-reporter=tap', import.meta.filename], {
        cwd: process.cwd(), encoding: 'utf8', env,
      });
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.match(result.stdout, /# tests 11\b/, result.stdout + result.stderr);
    });
  }
} else {
  const { default: config } = await import('../docs/.vitepress/config.ts');
  const docsRoot = path.resolve('docs');
  const md = await createMarkdownRenderer(docsRoot, config.markdown, config.base);
  const sourceChecker = await createSourceLinkChecker(path.dirname(docsRoot));
  const page = 'en/models/regression/lasso.md';
  function render(source, relativePath = page) {
    return md.render(source, {
      relativePath, path: path.join(docsRoot, relativePath), cleanUrls: true,
    });
  }
  function htmlLinks(html) {
    const result = [];
    function walk(node) {
      if (node.tagName === 'a') result.push(Object.fromEntries(node.attrs.map(attr => [attr.name, attr.value])));
      for (const child of node.childNodes ?? []) walk(child);
    }
    walk(parseFragment(html));
    return result;
  }

  test('nested repository references stay readable on GitHub with their fragment', () => {
    const html = render('[validation](../../../../dev/references/model-validation.md#lasso-and-elastic-net)');
    assert.match(html, /https:\/\/github\.com\/TheHiddenObserver\/statgpu\/blob\/master\/dev\/references\/model-validation\.md#lasso-and-elastic-net/);
  });

  test('canonical nested documentation and peer-language links remain local', () => {
    const html = render('[Ridge](ridge.md) [Chinese](../../../cn/models/regression/lasso.md)');
    assert.match(html, /href="\.\/ridge"/);
    assert.match(html, /href="\.\/\.\.\/\.\.\/\.\.\/cn\/models\/regression\/lasso"/);
    assert.doesNotMatch(html, /github\.com/);
  });

  test('survival directory anchors lead to the family overview without a standalone section', async () => {
    for (const [locale, anchors] of [
      ['en', ['survival-analysis', 'choosing-a-cox-estimator']],
      ['cn', ['生存分析', '如何选择-cox-估计器']],
    ]) {
      const relativePath = `${locale}/models/README.md`;
      const source = await readFile(path.join(docsRoot, relativePath), 'utf8');
      const html = render(source, relativePath);
      const rows = [];
      const ids = [];
      const headings = [];
      function walk(node, row = null) {
        if (node.tagName === 'tr') { row = { ids: [], links: [] }; rows.push(row); }
        const attrs = Object.fromEntries((node.attrs ?? []).map(attr => [attr.name, attr.value]));
        if (attrs.id) { ids.push(attrs.id); if (row) row.ids.push(attrs.id); }
        if (row && node.tagName === 'a' && attrs.href) row.links.push(attrs.href);
        if (/^h[1-6]$/.test(node.tagName ?? '')) headings.push(attrs.id);
        for (const child of node.childNodes ?? []) walk(child, row);
      }
      walk(parseFragment(html));
      for (const anchor of anchors) {
        assert.equal(ids.filter(id => id === anchor).length, 1, `${locale}: ${anchor}`);
        assert.ok(!headings.includes(anchor), `${locale}: no isolated survival section`);
        const row = rows.find(candidate => candidate.ids.includes(anchor));
        assert.ok(row?.links.includes('./survival/README'), `${locale}: old anchor reaches overview link`);
      }
      const overviewPath = `${locale}/models/survival/README.md`;
      const overview = await readFile(path.join(docsRoot, overviewPath), 'utf8');
      const links = htmlLinks(render(overview, overviewPath));
      assert.ok(links.some(link => link.href === './../coxph'));
      const counterpart = locale === 'en' ? 'cn' : 'en';
      assert.ok(links.some(link => link.href === `./../../../${counterpart}/models/survival/README`));
      const checked = await sourceChecker.checkDocument('docs/' + overviewPath, overview);
      assert.deepEqual(checked.errors, [], locale);
    }
  });

  test('dashboard entry bypasses docs routing and retains configured base', () => {
    const html = render('[dashboard](/dashboard/)');
    assert.ok(html.includes(`href="${config.base}dashboard/"`));
    assert.match(html, /target="_self"/);
  });

  test('all 40 maintained entry links exist in source and render to the intended website destinations', async () => {
    const generatedLinks = {
      'cn/README.md': ['dashboard/'],
      'cn/benchmarks.md': ['dashboard/'],
      'cn/index.md': ['dashboard/'],
      'cn/guides/benchmarks.md': ['dashboard/', 'dashboard/data/source_inventory.json', 'dashboard/data/parse_report.json', 'dashboard/data/benchmark_data.json'],
      'en/benchmarks.md': ['dashboard/'],
      'en/index.md': ['dashboard/'],
      'en/guides/benchmarks.md': ['dashboard/', 'dashboard/data/source_inventory.json', 'dashboard/data/parse_report.json', 'dashboard/data/benchmark_data.json'],
      'en/guides/statgpu_benchmark_dashboard.md': ['dashboard/', 'dashboard/data/source_inventory.json', 'dashboard/data/parse_report.json', 'dashboard/data/benchmark_data.json'],
      'index.md': ['dashboard/'],
    };
    const fragments = {
      'dashboard/': 'open-dashboard',
      'dashboard/data/source_inventory.json': 'source-inventory',
      'dashboard/data/parse_report.json': 'parse-report',
      'dashboard/data/benchmark_data.json': 'normalized-results',
    };
    let checked = 0;
    for (const [relativePath, routes] of Object.entries(generatedLinks)) {
      const source = await readFile(path.join(docsRoot, relativePath), 'utf8');
      const inspected = sourceChecker.inspect(source);
      const links = htmlLinks(render(source, relativePath));
      const result = await sourceChecker.checkDocument('docs/' + relativePath, source);
      assert.deepEqual(result.errors, [], relativePath);
      for (const route of routes) {
        const fallback = path.posix.relative(path.posix.dirname(relativePath), 'en/guides/statgpu_benchmark_dashboard.md') + '#' + fragments[route];
        assert.equal(inspected.links.filter(href => href === fallback).length, 1, `${relativePath}: ${fallback}`);
        const matching = links.filter(link => link.href === config.base + route);
        assert.equal(matching.length, 1, `${relativePath}: ${config.base + route}`);
        assert.equal(matching[0].target, '_self');
        checked += 1;
      }
    }
    const source = await readFile(path.join(docsRoot, 'index.md'), 'utf8');
    const inspected = sourceChecker.inspect(source);
    const links = htmlLinks(render(source, 'index.md'));
    const documentationTargets = [
      'en/models/README.md', 'en/guides/when-to-use-gpu.md',
      'en/guides/device-and-memory.md', 'en/getting-started/quickstart.md',
      'en/reference/linear-model-api.md#formula-inputs',
      'en/guides/cross-validation.md', 'en/guides/inference-modes.md',
      'en/guides/inference-api.md', 'en/models/regression/README.md',
      'en/models/survival/README.md', 'en/models/testing/README.md',
      'en/models/nonparametric/README.md', 'en/panel/README.md',
      'en/unsupervised/README.md', 'en/README.md#complete-api-references',
      'en/guides/implemented-methods.md', 'en/guides/solver-algorithms.md',
      'en/guides/solver-penalty-matrix.md', 'en/benchmarks.md',
      'en/acknowledgements.md', 'en/changelog.md', 'cn/README.md',
    ];
    for (const href of new Set(documentationTargets)) {
      const count = documentationTargets.filter(target => target === href).length;
      assert.equal(inspected.links.filter(target => target === href).length, count, href);
      assert.equal(links.filter(link => link.href === './' + href.replace(/\.md(?=#|$)/, '')).length, count, href);
    }
    assert.equal(checked + documentationTargets.length, 40);
  });

  test('ordinary dashboard-guide and section links are not rewritten to generated resources', () => {
    const links = htmlLinks(render('[guide](statgpu_benchmark_dashboard.md) [coverage](statgpu_benchmark_dashboard.md#coverage)', 'en/guides/benchmarks.md'));
    assert.deepEqual(links.map(link => link.href), ['./statgpu_benchmark_dashboard', './statgpu_benchmark_dashboard#coverage']);
  });

  test('generated JSON links preserve queries and do not gain Markdown HTML suffixes', () => {
    const html = render('[inventory](statgpu_benchmark_dashboard.md?download=1#source-inventory)', 'en/guides/benchmarks.md');
    assert.ok(html.includes(`href="${config.base}dashboard/data/source_inventory.json?download=1"`));
    assert.doesNotMatch(html, /\.json\.html|statgpu\/statgpu/);
  });

  test('deleted fallback pages and anchors fail the renderer instead of hiding behind generated output', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'statgpu-fallback-'));
    try {
      const fixtureDocs = path.join(root, 'docs');
      const fixtureGuide = path.join(fixtureDocs, 'en/guides/statgpu_benchmark_dashboard.md');
      await mkdir(path.dirname(fixtureGuide), { recursive: true });
      const fixture = new md.constructor();
      installSourceLinks(fixture, { docsRoot: fixtureDocs, base: config.base });
      const env = { relativePath: 'index.md' };
      const link = '[inventory](en/guides/statgpu_benchmark_dashboard.md#source-inventory)';
      assert.throws(() => fixture.render(link, env), /Invalid generated-resource source fallback/);
      await writeFile(fixtureGuide, '# Guide without fallback anchors\n');
      assert.throws(() => fixture.render(link, env), /Invalid generated-resource source fallback/);
      await writeFile(fixtureGuide, '<a id="source-inventory"></a>\n');
      assert.ok(fixture.render(link, env).includes(`href="${config.base}dashboard/data/source_inventory.json"`));
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test('archived Markdown is linked as a repository artifact', () => {
    const html = render('[archive](changelog-history-through-2026-08-03.markdown)', 'en/changelog.md');
    assert.match(html, /blob\/master\/docs\/en\/changelog-history-through-2026-08-03\.markdown/);
  });

  test('outside-repository and nonexistent repository targets fail the build', () => {
    assert.throws(() => render('[escape](../../../../../../etc/passwd)'), /Invalid repository link/);
    assert.throws(() => render('[missing](../../../../dev/references/no-such-page.md)'), /Invalid repository link/);
  });

  test('ordinary external URLs and same-page fragments retain their destination', () => {
    const html = render('[external](https://example.org/reference#part) [section](#why-use-lasso)');
    assert.match(html, /href="https:\/\/example\.org\/reference#part"/);
    assert.match(html, /href="#why-use-lasso"/);
  });
}
