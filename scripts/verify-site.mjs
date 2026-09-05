import { readFile, readdir, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
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

const documentationSections = ['models', 'panel', 'unsupervised'];
const documentationSourceFailures = [];
const untranslatedChineseHeadings = new Set([
  'Overview',
  'Path',
  'Estimator',
  'Parameters',
  'Outputs',
  'FAQ',
  'References',
  'External Validation',
  'CPU and GPU Example',
  'Formula Example',
  'Statistical Model and Identification',
  'Statistical Model and Target',
  'Covariance and Inference',
  'Numerical and Strict Behavior',
  'Strict/Approx Difference',
]);

function markdownOutline(source) {
  const headings = [];
  let fenceCount = 0;
  let inFence = false;
  for (const line of source.split(/\r?\n/)) {
    if (/^\s*(?:```|~~~)/.test(line)) {
      fenceCount += 1;
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const match = /^(#{1,3})\s+(.+?)\s*$/.exec(line);
    if (match) {
      headings.push({ level: match[1].length, title: match[2] });
    }
  }
  return { headings, fenceCount, hasOpenFence: inFence };
}

for (const section of documentationSections) {
  const englishDirectory = path.join(repoRoot, 'docs', 'en', section);
  const chineseDirectory = path.join(repoRoot, 'docs', 'cn', section);
  const englishNames = (await readdir(englishDirectory, { withFileTypes: true }))
    .filter(entry => entry.isFile() && entry.name.endsWith('.md'))
    .map(entry => entry.name)
    .sort();
  const chineseNames = (await readdir(chineseDirectory, { withFileTypes: true }))
    .filter(entry => entry.isFile() && entry.name.endsWith('.md'))
    .map(entry => entry.name)
    .sort();

  if (JSON.stringify(englishNames) !== JSON.stringify(chineseNames)) {
    documentationSourceFailures.push(
      `docs/${section}: English and Chinese Markdown file sets differ`,
    );
  }

  for (const name of englishNames) {
    if (!chineseNames.includes(name)) continue;
    const englishSource = await readFile(
      path.join(englishDirectory, name),
      'utf8',
    );
    const chineseSource = await readFile(
      path.join(chineseDirectory, name),
      'utf8',
    );
    const englishOutline = markdownOutline(englishSource);
    const chineseOutline = markdownOutline(chineseSource);
    const englishH1 = englishOutline.headings.find(heading => heading.level === 1);
    const chineseH1 = chineseOutline.headings.find(heading => heading.level === 1);
    const pageLabel = `${section}/${name}`;

    if (!englishH1 || /[\u3400-\u9fff]/u.test(englishH1.title)) {
      documentationSourceFailures.push(
        `docs/en/${pageLabel}: expected an English H1`,
      );
    }
    if (!chineseH1 || !/[\u3400-\u9fff]/u.test(chineseH1.title)) {
      documentationSourceFailures.push(
        `docs/cn/${pageLabel}: expected a Chinese H1`,
      );
    }
    for (const level of [2, 3]) {
      const englishCount = englishOutline.headings.filter(
        heading => heading.level === level,
      ).length;
      const chineseCount = chineseOutline.headings.filter(
        heading => heading.level === level,
      ).length;
      if (englishCount !== chineseCount) {
        documentationSourceFailures.push(
          `docs/${pageLabel}: H${level} count differs (${englishCount}/${chineseCount})`,
        );
      }
    }
    if (
      englishOutline.fenceCount !== chineseOutline.fenceCount ||
      englishOutline.hasOpenFence ||
      chineseOutline.hasOpenFence
    ) {
      documentationSourceFailures.push(
        `docs/${pageLabel}: fenced-code structure differs or is unclosed`,
      );
    }
    for (const heading of chineseOutline.headings) {
      if (untranslatedChineseHeadings.has(heading.title)) {
        documentationSourceFailures.push(
          `docs/cn/${pageLabel}: untranslated heading "${heading.title}"`,
        );
      }
    }

    const chineseTarget =
      name === 'index.md'
        ? `../../cn/${section}/`
        : `../../cn/${section}/${name}`;
    const englishTarget =
      name === 'index.md'
        ? `../../en/${section}/`
        : `../../en/${section}/${name}`;
    if (!englishSource.includes(`](${chineseTarget})`)) {
      documentationSourceFailures.push(
        `docs/en/${pageLabel}: missing Chinese page switch`,
      );
    }
    if (!chineseSource.includes(`](${englishTarget})`)) {
      documentationSourceFailures.push(
        `docs/cn/${pageLabel}: missing English page switch`,
      );
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

function resolveLocalTarget(htmlFile, rawTarget) {
  const withoutHash = rawTarget.split('#', 1)[0].split('?', 1)[0];
  if (!withoutHash) return null;
  if (/^(?:[a-z]+:|\/\/)/i.test(withoutHash)) return null;

  let candidate;
  if (withoutHash.startsWith('/')) {
    if (!withoutHash.startsWith(siteBase)) {
      throw new Error(
        `Root-absolute URL is outside STATGPU_SITE_BASE: ${rawTarget}`,
      );
    }
    candidate = path.join(siteDir, withoutHash.slice(siteBase.length));
  } else {
    candidate = path.resolve(path.dirname(htmlFile), withoutHash);
  }
  const relative = path.relative(siteDir, candidate);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Local URL escapes the deployment artifact: ${rawTarget}`);
  }
  return candidate;
}

function targetExists(candidate) {
  return (
    indexedFileExists(candidate) ||
    indexedFileExists(`${candidate}.html`) ||
    indexedFileExists(path.join(candidate, 'index.html'))
  );
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
const attributePattern = /\b(?:href|src)=(['"])(.*?)\1/g;
for (const htmlFile of htmlFiles) {
  const html = await readFile(htmlFile, 'utf8');
  for (const match of html.matchAll(attributePattern)) {
    const target = match[2];
    if (
      !target ||
      target.startsWith('#') ||
      target.startsWith('data:') ||
      target.startsWith('mailto:') ||
      target.startsWith('tel:') ||
      target.startsWith('javascript:')
    ) {
      continue;
    }
    if (target.includes('docs/assets/benchmarks')) {
      failures.push(
        `${path.relative(siteDir, htmlFile)} -> ${target}: legacy dashboard path`,
      );
      continue;
    }
    try {
      const resolved = resolveLocalTarget(htmlFile, target);
      if (resolved && !targetExists(resolved)) {
        failures.push(
          `${path.relative(siteDir, htmlFile)} -> ${target}`,
        );
      }
    } catch (error) {
      failures.push(
        `${path.relative(siteDir, htmlFile)} -> ${target}: ${error.message}`,
      );
    }
  }
}

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
const searchIndexBytes = (
  await Promise.all(searchIndexFiles.map(async file => (await stat(file)).size))
).reduce((total, size) => total + size, 0);
const searchIndexContents = await Promise.all(
  searchIndexFiles.map(file => readFile(file, 'utf8')),
);
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
    `local search indexes exceed 1.6 MiB: ${searchIndexBytes} bytes`,
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
]) {
  if (!combinedSearchIndex.includes(requiredPath)) {
    failures.push(`local search index is missing user guide: ${requiredPath}`);
  }
}

if (failures.length) {
  throw new Error(`Site verification failed:\n${failures.join('\n')}`);
}
console.log(
  `Verified ${htmlFiles.length} HTML pages at base ${siteBase}; dashboard JS ${dashboardJsBytes} bytes; benchmark data ${benchmarkBytes} bytes; search indexes ${searchIndexBytes} bytes`,
);
