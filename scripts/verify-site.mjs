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

const allFiles = await filesBelow(siteDir);
const fileIndex = new Set(
  allFiles.map(file => path.resolve(file).toLocaleLowerCase('en-US')),
);
function indexedFileExists(file) {
  return fileIndex.has(path.resolve(file).toLocaleLowerCase('en-US'));
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
const chineseHome = await readFile(path.join(siteDir, 'cn', 'index.html'), 'utf8');
if (!chineseHome.includes('<html lang="zh-CN"')) {
  failures.push('cn/index.html: expected html lang="zh-CN"');
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

const dashboardAssets = path
  .join(siteDir, 'dashboard', 'assets')
  .toLocaleLowerCase('en-US');
const dashboardJs = allFiles.filter(
  file =>
    file.toLocaleLowerCase('en-US').startsWith(dashboardAssets) &&
    file.endsWith('.js'),
);
const dashboardJsBytes = (
  await Promise.all(dashboardJs.map(async file => (await stat(file)).size))
).reduce((total, size) => total + size, 0);
const benchmarkBytes = (
  await stat(path.join(siteDir, 'dashboard', 'data', 'benchmark_data.json'))
).size;
if (dashboardJsBytes > 750 * 1024) {
  failures.push(`dashboard JavaScript exceeds 750 KiB: ${dashboardJsBytes} bytes`);
}
if (benchmarkBytes > 6 * 1024 * 1024) {
  failures.push(`benchmark_data.json exceeds 6 MiB: ${benchmarkBytes} bytes`);
}

if (failures.length) {
  throw new Error(`Site verification failed:\n${failures.join('\n')}`);
}
console.log(
  `Verified ${htmlFiles.length} HTML pages at base ${siteBase}; dashboard JS ${dashboardJsBytes} bytes; benchmark data ${benchmarkBytes} bytes`,
);
