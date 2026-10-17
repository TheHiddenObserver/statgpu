import { readFile, readdir, stat } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { decodeSearchIndex } from './search-index-compression.mjs';
import { validateSiteHtml } from './site-html-validation.mjs';
import path from 'node:path';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const siteDir = path.join(repoRoot, '.site-dist');
const rawBase = process.env.STATGPU_SITE_BASE?.trim() || '/statgpu/';
if (!rawBase.startsWith('/') || !rawBase.endsWith('/')) {
  throw new Error('STATGPU_SITE_BASE must start and end with "/"');
}
const siteBase = rawBase.replace(/\/{2,}/g, '/');

async function filesBelow(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await filesBelow(absolute)));
    else if (entry.isFile()) files.push(absolute);
  }
  return files;
}

// Current model pages are nested by family. Verify the complete recursive
// bilingual source inventory and page switches, without requiring translations
// to duplicate the exact heading counts of the English prose.
const documentationSourceFailures = [];
for (const section of ['models', 'panel', 'unsupervised', 'reference']) {
  const roots = ['en', 'cn'].map(locale => path.join(repoRoot, 'docs', locale, section));
  const inventories = await Promise.all(roots.map(async root =>
    (await filesBelow(root)).filter(file => file.endsWith('.md')).map(file => path.relative(root, file)).sort(),
  ));
  if (JSON.stringify(inventories[0]) !== JSON.stringify(inventories[1])) {
    documentationSourceFailures.push(`${section}: bilingual Markdown inventories differ`);
  }
  for (const [index, locale] of ['en', 'cn'].entries()) {
    for (const name of inventories[index]) {
      const file = path.join(roots[index], name);
      const source = await readFile(file, 'utf8');
      const counterpart = path.join(roots[1 - index], name);
      const peer = path.relative(path.dirname(file), counterpart).replace(/\\/g, '/');
      const label = `docs/${locale}/${section}/${name}`;
      if (!/^# .+/m.test(source)) documentationSourceFailures.push(`${label}: missing H1`);
      if (locale === 'cn' && !/[\u3400-\u9fff]/u.test(source)) {
        documentationSourceFailures.push(`${label}: missing Chinese content`);
      }
      if (!source.includes(`](${peer})`)) {
        documentationSourceFailures.push(`${label}: missing counterpart link ${peer}`);
      }
      let fence = null;
      for (const line of source.split(/\r?\n/)) {
        const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
        if (!marker) continue;
        if (fence === null) fence = marker;
        else if (marker[0] === fence[0] && marker.length >= fence.length) fence = null;
      }
      if (fence !== null) documentationSourceFailures.push(`${label}: unclosed code fence`);
    }
  }
}

const allFiles = await filesBelow(siteDir);
// GitHub Pages paths are case-sensitive. Index exact resolved paths so the
// verifier cannot accept /Foo when the deployed artifact contains only foo.
const fileIndex = new Set(allFiles.map(file => path.resolve(file)));
function indexedFileExists(file) {
  return fileIndex.has(path.resolve(file));
}

const required = [
  'index.html',
  'en/index.html',
  'cn/index.html',
  'dashboard/index.html',
  'dashboard/data/benchmark_data.json',
  'dashboard/data/parse_report.json',
  'dashboard/data/source_inventory.json',
  '.nojekyll',
];
for (const relative of required) {
  if (!indexedFileExists(path.join(siteDir, relative))) {
    throw new Error(`Missing required deployment file: ${relative}`);
  }
}

const htmlFiles = allFiles.filter(file => file.endsWith('.html'));
const failures = [];
failures.push(...documentationSourceFailures);
const chineseHome = await readFile(path.join(siteDir, 'cn', 'index.html'), 'utf8');
if (!chineseHome.includes('<html lang="zh-CN"')) {
  failures.push('cn/index.html: expected html lang="zh-CN"');
}
const mathProbe = await readFile(
  path.join(siteDir, 'en', 'panel', 'random-effects.html'),
  'utf8',
);
if (!mathProbe.includes('<mjx-container')) {
  failures.push('en/panel/random-effects.html: expected rendered MathJax markup');
}
failures.push(...await validateSiteHtml({ siteDir, siteBase, allFiles }));

const dashboardAssets = path.resolve(siteDir, 'dashboard', 'assets');
const dashboardJs = allFiles.filter(
  file => path.resolve(file).startsWith(dashboardAssets + path.sep) && file.endsWith('.js'),
);
const dashboardJsBytes = (
  await Promise.all(dashboardJs.map(async file => (await stat(file)).size))
).reduce((total, size) => total + size, 0);
const benchmarkBytes = (
  await stat(path.join(siteDir, 'dashboard', 'data', 'benchmark_data.json'))
).size;
const searchIndexFiles = allFiles.filter(
  file =>
    path.basename(file).startsWith('@localSearchIndex') &&
    file.endsWith('.js'),
);
const searchDecoderFiles = allFiles.filter(
  file => path.basename(file).startsWith('search-index-decoder.') && file.endsWith('.js'),
);
const searchIndexBytes = (
  await Promise.all([...searchIndexFiles, ...searchDecoderFiles].map(async file => (await stat(file)).size))
).reduce((total, size) => total + size, 0);
if (searchDecoderFiles.length !== 1) {
  failures.push(`expected one measurable search decoder, found ${searchDecoderFiles.length}`);
}
const searchIndexContents = await Promise.all(searchIndexFiles.map(async file => {
  const { default: payload } = await import(pathToFileURL(file).href);
  const serialized = await decodeSearchIndex(payload);
  if (createHash('sha256').update(serialized).digest('hex') !== payload.sha256) {
    failures.push(`${path.basename(file)}: decoded search index checksum mismatch`);
  }
  const parsed = JSON.parse(serialized);
  if (!Number.isSafeInteger(parsed.documentCount) || parsed.documentCount <= 0 || !Array.isArray(parsed.index)) {
    failures.push(`${path.basename(file)}: empty or invalid decoded search index`);
  }
  return serialized;
}));
const combinedSearchIndex = searchIndexContents.join('\n');

if (dashboardJsBytes > 750 * 1024) {
  failures.push(`dashboard JavaScript exceeds 750 KiB: ${dashboardJsBytes} bytes`);
}
if (benchmarkBytes > 6 * 1024 * 1024) {
  failures.push(`benchmark_data.json exceeds 6 MiB: ${benchmarkBytes} bytes`);
}
if (searchIndexFiles.length !== 2) {
  failures.push(
    `expected two locale search indexes, found ${searchIndexFiles.length}`,
  );
}
if (searchIndexBytes > 1.6 * 1024 * 1024) {
  failures.push(
    `local search modules and decoder exceed 1.6 MiB: ${searchIndexBytes} bytes`,
  );
}
for (const excludedPath of [
  '/benchmark-dashboard/',
  '/en/benchmarks#',
  '/cn/benchmarks#',
  '/en/changelog#',
  '/cn/changelog#',
  'changelog-history-through-',
  '/website-deployment#',
]) {
  if (combinedSearchIndex.includes(excludedPath)) {
    failures.push(`local search index contains excluded path: ${excludedPath}`);
  }
}
for (const requiredPath of [
  '/en/guides/solver-algorithms',
  '/cn/guides/solver-algorithms',
  '/en/reference/linear-model-api',
  '/cn/reference/linear-model-api',
]) {
  if (!combinedSearchIndex.includes(requiredPath)) {
    failures.push(`local search index is missing public documentation: ${requiredPath}`);
  }
}

if (failures.length) {
  throw new Error(`Site verification failed:\n${failures.join('\n')}`);
}
console.log(
  `Verified ${htmlFiles.length} HTML pages at base ${siteBase}; dashboard JS ${dashboardJsBytes} bytes; benchmark data ${benchmarkBytes} bytes; search modules and decoder ${searchIndexBytes} bytes`,
);
