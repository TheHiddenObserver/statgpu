import { access, cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const siteDir = path.join(repoRoot, '.site-dist');
const dashboardDist = path.join(repoRoot, 'frontend', 'dist');
const dashboardTarget = path.join(siteDir, 'dashboard');

async function requireFile(file) {
  try {
    await access(file, constants.R_OK);
  } catch {
    throw new Error(`Required build input is missing: ${path.relative(repoRoot, file)}`);
  }
}

await requireFile(path.join(siteDir, 'index.html'));
await requireFile(path.join(dashboardDist, 'index.html'));
await requireFile(path.join(dashboardDist, 'data', 'benchmark_data.json'));

await rm(dashboardTarget, { recursive: true, force: true });
await mkdir(dashboardTarget, { recursive: true });
await cp(dashboardDist, dashboardTarget, { recursive: true });
await writeFile(path.join(siteDir, '.nojekyll'), '', 'utf8');

console.log('Assembled VitePress and dashboard output in .site-dist');
