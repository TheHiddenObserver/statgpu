import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const siteDir = path.join(repoRoot, '.site-dist');

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

const digest = createHash('sha256');
for (const file of (await filesBelow(siteDir)).sort()) {
  digest.update(path.relative(siteDir, file).replaceAll(path.sep, '/'));
  digest.update('\0');
  digest.update(await readFile(file));
  digest.update('\0');
}
console.log(digest.digest('hex'));
