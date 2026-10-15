import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cp, lstat, mkdtemp, readFile, rm, symlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const generatedDirectories = [
  'frontend/public/data', 'frontend/dist', '.site-dist',
  'docs/assets/benchmarks', 'docs/.vitepress/cache', 'docs/.vitepress/.temp',
];
const bundleNames = ['benchmark_data.json', 'parse_report.json', 'source_inventory.json'];

async function documentedBuildCommands() {
  const readme = await readFile(path.join(repositoryRoot, 'frontend/README.md'), 'utf8');
  const section = readme.split('## Production build and staleness\n')[1]?.split('\n## ')[0];
  const recipe = section?.match(/```bash\n([\s\S]*?)\n```/)?.[1];
  assert.ok(recipe, 'The production section must include a standalone build recipe');
  // CI installs these exact lockfiles before invoking this test. Reuse only
  // those dependencies, never generated data or previously built assets.
  assert.match(recipe, /^npm ci\nnpm ci --prefix frontend\n/);
  const commands = recipe.split(/(?<!\\)\n/).filter(command => command.trim());
  assert.deepEqual(commands.splice(0, 2), ['npm ci', 'npm ci --prefix frontend']);
  assert.equal(commands.at(-1), 'npm run site:build');
  const generators = commands.filter(command => command.includes('generate_benchmark_data.py'));
  assert.equal(generators.length, 1, 'Generate the bundle explicitly before the build');
  assert.match(generators[0], /--deterministic\b/);
  assert.match(generators[0], /--strict-sources\b/);
  return { commands, generator: generators[0] };
}

async function cleanCheckout() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'statgpu-production-recipe-'));
  await cp(repositoryRoot, root, {
    recursive: true,
    filter(source) {
      const relative = path.relative(repositoryRoot, source).replaceAll(path.sep, '/');
      const parts = relative.split('/');
      return !parts.some(part => ['.git', 'node_modules', '__pycache__', '.pytest_cache'].includes(part))
        && !generatedDirectories.some(directory => relative === directory || relative.startsWith(directory + '/'));
    },
  });
  for (const directory of generatedDirectories) {
    await assert.rejects(lstat(path.join(root, directory)), { code: 'ENOENT' });
  }
  for (const directory of ['node_modules', 'frontend/node_modules']) {
    const installed = path.join(repositoryRoot, directory);
    await lstat(installed);
    await symlink(installed, path.join(root, directory), 'junction');
  }
  return root;
}

function runRecipe(root, commands, base) {
  const env = {
    ...process.env,
    STATGPU_SITE_BASE: base,
    STATGPU_SITE_URL: base === '/' ? 'https://statgpu.example/' : 'https://thehiddenobserver.github.io/statgpu/',
  };
  // No mocks: execute the documented generator and full build, including
  // VitePress, TypeScript/Vite, assembly, HTML validation, and size budgets.
  return spawnSync('bash', ['-e', '-o', 'pipefail', '-c', commands.join('\n')], {
    cwd: root, env, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
    timeout: 300_000,
  });
}

async function bundleIdentity(root) {
  const hashes = {};
  const ids = [];
  for (const name of bundleNames) {
    const source = await readFile(path.join(root, 'frontend/public/data', name));
    for (const directory of ['frontend/dist/data', '.site-dist/dashboard/data']) {
      const copy = await readFile(path.join(root, directory, name));
      if (directory === '.site-dist/dashboard/data' && name === 'benchmark_data.json') {
        // Assembly intentionally minifies the main payload; its data must be
        // identical without requiring the source's pretty-print whitespace.
        assert.deepEqual(JSON.parse(copy), JSON.parse(source));
      } else {
        assert.deepEqual(copy, source);
      }
    }
    hashes[name] = createHash('sha256').update(source).digest('hex');
    const data = JSON.parse(source);
    ids.push(data.generation_id ?? data.meta?.generation_id);
  }
  assert.ok(ids[0]);
  assert.deepEqual(ids, [ids[0], ids[0], ids[0]]);
  return { hashes, generationId: ids[0] };
}

test('documented production recipe builds from source without generated files', { timeout: 1_200_000 }, async t => {
  const { commands, generator } = await documentedBuildCommands();
  let projectHash;
  let firstBundle;
  for (const [label, base, omitGeneration] of [
    ['missing-generation negative control', '/statgpu/', true],
    ['project base', '/statgpu/', false],
    ['custom-domain root base', '/', false],
    ['repeated project base', '/statgpu/', false],
  ]) {
    await t.test(label, async () => {
      const root = await cleanCheckout();
      try {
        const selected = omitGeneration ? commands.filter(command => command !== generator) : commands;
        const result = runRecipe(root, selected, base);
        assert.ifError(result.error);
        const output = result.stdout + result.stderr;
        if (omitGeneration) {
          assert.notEqual(result.status, 0, output);
          assert.match(output, /Required build input is missing: frontend\/dist\/data\/benchmark_data\.json/);
          t.diagnostic('Omitting generation reaches assembly and fails on missing benchmark_data.json');
          return;
        }
        assert.equal(result.status, 0, output);
        assert.match(output, /Verified \d+ HTML/);
        const bundle = await bundleIdentity(root);
        firstBundle ??= bundle;
        assert.deepEqual(bundle, firstBundle, 'Deterministic bundles must match across clean builds and bases');
        const hash = spawnSync(process.execPath, ['scripts/hash-site.mjs'], { cwd: root, encoding: 'utf8' });
        assert.equal(hash.status, 0, hash.stdout + hash.stderr);
        const digest = hash.stdout.trim();
        assert.match(digest, /^[a-f0-9]{64}$/);
        if (base === '/statgpu/') {
          projectHash ??= digest;
          assert.equal(digest, projectHash, 'Clean project-base builds must be byte-for-byte reproducible');
        }
        t.diagnostic(`${label}: ${bundle.generationId}; site SHA256 ${digest}`);
        t.diagnostic(output);
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });
  }
});
