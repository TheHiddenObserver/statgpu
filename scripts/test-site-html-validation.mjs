import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createMarkdownRenderer } from 'vitepress';
import config from '../docs/.vitepress/config.ts';
import { inspectHtml, validateSiteHtml } from './site-html-validation.mjs';

const docsRoot = path.resolve('docs');
const md = await createMarkdownRenderer(docsRoot, config.markdown, config.base);
function render(source, relativePath = 'en/reference/linear-model-api.md') {
  return md.render(source, { relativePath, path: path.join(docsRoot, relativePath), cleanUrls: true });
}
async function fixture(t, files, base = '/statgpu/') {
  const siteDir = await mkdtemp(path.join(os.tmpdir(), 'statgpu-site-html-'));
  t.after(() => rm(siteDir, { recursive: true, force: true }));
  const allFiles = [];
  for (const [name, source] of Object.entries(files)) {
    const file = path.join(siteDir, name);
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, source);
    allFiles.push(file);
  }
  return validateSiteHtml({ siteDir, siteBase: base, allFiles });
}

test('real renderer preserves source identifiers and compatible VitePress aliases', () => {
  const pairs = [
    ['PenalizedGLM_CV', 'penalizedglm_cv', 'penalizedglm-cv'],
    ['bootstrap_statistic', 'bootstrap_statistic', 'bootstrap-statistic'],
    ['1. Proximal IRLS-CD', '1-proximal-irls-cd', '_1-proximal-irls-cd'],
    ['系数、推断与 CV 结果', '系数推断与-cv-结果', '系数、推断与-cv-结果'],
    ['进阶：EM 更新公式', '进阶em-更新公式', '进阶-em-更新公式'],
    ['Covariance/inference', 'covarianceinference', 'covariance-inference'],
  ];
  for (const [heading, sourceId, siteId] of pairs) {
    const html = render(`## ${heading}\n\n[Source](#${sourceId}) [Site](#${siteId})`);
    const { ids, duplicateIds } = inspectHtml(html);
    assert.ok(ids.has(sourceId), `${heading}: source ID ${sourceId}`);
    assert.ok(ids.has(siteId), `${heading}: VitePress ID ${siteId}`);
    assert.equal(duplicateIds.size, 0);
    assert.ok(html.includes(`<h2 id="${sourceId}"`));
    assert.ok(html.includes(`class="header-anchor" href="#${sourceId}"`));
  }
});

test('source IDs win cross-policy collisions, repetitions remain unique and state resets per page', () => {
  const source = '## A_B\n\n## A-B\n\n## A_B\n\n## A B-1\n';
  const html = render(source);
  const headings = [...html.matchAll(/<h2 id="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(headings, ['a_b', 'a-b', 'a_b-1', 'a-b-1']);
  assert.equal(inspectHtml(html).duplicateIds.size, 0);
  assert.equal(render(source), html);
  // The legacy a-b alias for the first heading would steal the second
  // heading's source fragment. It must never be attached to the first.
  assert.doesNotMatch(html, /<h2 id="a_b"[^>]*><span id="a-b"/);
});

test('manual anchors and explicit heading IDs are reserved before aliases are assigned', () => {
  const html = render('<a id="a_b"></a>\n\n## A_B\n\n## Custom {#a-b}\n\n<a id="later"></a>\n');
  assert.equal(inspectHtml(html).duplicateIds.size, 0);
  assert.match(html, /<h2 id="a_b-1"/);
  assert.match(html, /<h2 id="a-b"/);
  assert.ok(inspectHtml(html).ids.has('later'));
});

test('the unchanged old renderer fails the final-HTML fragment check', async t => {
  let html;
  md.core.ruler.disable('statgpu-heading-ids');
  try {
    html = render('## PenalizedGLM_CV\n\n[Reference](#penalizedglm_cv)');
  } finally {
    md.core.ruler.enable('statgpu-heading-ids');
  }
  const failures = await fixture(t, { 'index.html': html });
  assert.equal(failures.length, 1);
  assert.match(failures[0], /missing HTML fragment "penalizedglm_cv"/);
});

for (const base of ['/statgpu/', '/']) {
  test(`final HTML validates same-page, cross-page, clean URLs and Unicode at base ${base}`, async t => {
    const failures = await fixture(t, {
      'index.html': `<a href="#home">same</a><h1 id=home>Home</h1>
        <a href="${base}en/reference?mode=1&amp;lang=cn#%E5%8F%82%E6%95%B0">Unicode</a>
        <a href="./en/reference.html#x%26y">entity</a>
        <a href="./en/#named">named</a><a href="${base}dashboard/#dashboard-main">dashboard</a>
        <script>const example = '<a href="#does-not-exist">';</script>
        <!-- <a href="#comment"> --><a href="https://example.org/#external">external</a>`,
      'en/reference.html': '<h2 id="参数">参数</h2><span id="x&amp;y"></span>',
      'en/index.html': '<a name="named"></a>',
      'dashboard/index.html': '<a href="#dashboard-main">Skip</a><main id="dashboard-main" tabindex="-1"></main>',
    }, base);
    assert.deepEqual(failures, []);
  });
}

test('missing same-page and cross-page fragments fail, including dashboard fragments', async t => {
  const failures = await fixture(t, {
    'index.html': '<a href="#missing">same</a><a href="./other#absent">other</a><a href="./dashboard/#bad">dashboard</a>',
    'other.html': '<h2 id="present">Section</h2>',
    'dashboard/index.html': '<a href="#dashboard-main">Skip</a><div id="app"></div>',
  });
  assert.equal(failures.length, 4);
  for (const fragment of ['missing', 'absent', 'bad', 'dashboard-main']) {
    assert.ok(failures.some(failure => failure.includes(`fragment "${fragment}"`)));
  }
});

test('path errors, malformed escaping, duplicates and inert template targets fail', async t => {
  const failures = await fixture(t, {
    'index.html': '<a href="/outside#x">base</a><a href="./Missing#x">case</a><a href="./%zz#x">encoding</a><a href="#inert">template</a><template><h2 id="inert"></h2></template><p id="same"></p><p id="same"></p>',
    'missing.html': '<h2 id="x">Present</h2>',
  });
  assert.equal(failures.length, 5);
  assert.ok(failures.some(failure => failure.includes('outside STATGPU_SITE_BASE')));
  assert.ok(failures.some(failure => failure.includes('missing deployment file')));
  assert.ok(failures.some(failure => failure.includes('URI malformed')));
  assert.ok(failures.some(failure => failure.includes('duplicate HTML id "same"')));
  assert.ok(failures.some(failure => failure.includes('fragment "inert"')));
});

test('real MathJax errors are rejected beyond the original probe page', async t => {
  const html = render('$$\n\\lceil n\\,\\text{support_fraction}\\rceil\n$$\n\n$$\n\\texttt{mean_diff}=\\bar y_i-\\bar y_j.\n$$');
  assert.match(html, /<mjx-container/);
  const failures = await fixture(t, {
    'en/panel/random-effects.html': render('$$x^2$$'),
    'cn/models/covariance/covariance-min-cov-det.html': html,
  });
  // SVG and assistive MathML each contain an error node for both equations.
  assert.equal(failures.filter(failure => failure.includes('MathJax error')).length, 4);
  assert.ok(failures.every(failure => failure.startsWith('cn/models/covariance/')));
});

test('MathML and CHTML error nodes are rejected without relying on one attribute spelling', async t => {
  const failures = await fixture(t, {
    'index.html': '<math><merror><mtext>bad</mtext></merror></math><mjx-merror>bad</mjx-merror><svg><g data-mml-node="merror"></g></svg>',
  });
  assert.equal(failures.length, 3);
  assert.ok(failures.every(failure => failure.includes('MathJax error')));
});

test('all actual Markdown pages render without duplicate IDs or MathJax errors', async () => {
  async function markdownFiles(dir) {
    const entries = await readdir(dir, { withFileTypes: true });
    const result = [];
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) result.push(...await markdownFiles(file));
      else if (file.endsWith('.md')) result.push(file);
    }
    return result;
  }
  const files = await markdownFiles(docsRoot);
  assert.ok(files.length > 200);
  for (const file of files) {
    const relative = path.relative(docsRoot, file);
    const html = render(await readFile(file, 'utf8'), relative);
    const result = inspectHtml(html);
    assert.deepEqual([...result.duplicateIds], [], `${relative}: duplicate IDs`);
    assert.deepEqual(result.mathErrors, [], `${relative}: MathJax errors`);
  }
});
