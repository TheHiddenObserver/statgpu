import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { generatedArtifactRoots } from './check-tracked-artifacts.mjs';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const checker = 'scripts/check-tracked-artifacts.mjs';
const tests = 'scripts/test-tracked-artifacts.mjs';
const workflowPath = '.github/workflows/benchmark-frontend.yml';
// Deliberately independent of the implementation: dropping a policy root must
// fail a test, rather than silently removing its negative control too.
const forbiddenRoots = [
  'node_modules', 'frontend/node_modules', 'frontend/test-results',
  'frontend/playwright-report', 'frontend/dist', 'frontend/public/data',
  'docs/assets/benchmarks', 'docs/.vitepress/cache', 'docs/.vitepress/.temp', '.site-dist',
];
const guardName = 'Reject tracked generated website artifacts';
const cleanName = 'Verify repository tree remains clean';
const afterQaName = 'Verify repository tree remains clean after browser QA';
const cleanCommands = ['git diff --exit-code', 'test -z "$(git status --porcelain --untracked-files=all)"'];

function run(root, program, args) {
  // Fixture Git repositories must not inherit a caller's alternate index or
  // worktree. No Git commits, identities, or hooks are needed for these tests.
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('GIT_')));
  return spawnSync(program, args, { cwd: root, env, encoding: 'utf8', timeout: 30_000 });
}

function git(root, ...args) {
  const result = run(root, 'git', args);
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  return result.stdout;
}

async function write(root, filename, content = 'fixture\n') {
  const fullPath = path.join(root, filename);
  await mkdir(path.dirname(fullPath), { recursive: true });
  await writeFile(fullPath, content);
}

async function fixture(t, files = []) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'statgpu-tracked-artifacts-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  git(root, 'init', '--quiet');
  git(root, 'config', 'core.autocrlf', 'false');
  await write(root, '.gitignore', forbiddenRoots.map(directory => directory + '/').join('\n') + '\n');
  await write(root, 'README.md');
  await mkdir(path.join(root, 'scripts'));
  await copyFile(path.join(repositoryRoot, checker), path.join(root, checker));
  for (const filename of files) await write(root, filename);
  git(root, 'add', '--force', '--all');
  await snapshot(root);
  return root;
}

async function snapshot(root) {
  // A tree-backed detached HEAD gives Git diff/status the same tracked-file
  // baseline as a clean committed checkout, without creating a commit object
  // or modifying any target-repository ref/index. This is a temporary fixture,
  // not a claim that the tree object is a normal repository commit.
  const tree = git(root, 'write-tree').trim();
  await writeFile(path.join(root, '.git/HEAD'), tree + '\n');
  assert.equal(git(root, 'cat-file', '-t', 'HEAD').trim(), 'tree');
  assertClean(root);
}

function assertClean(root) {
  assert.equal(git(root, 'diff', '--exit-code'), '');
  assert.equal(git(root, 'status', '--porcelain', '--untracked-files=all'), '');
}

function check(root, cwd = root) {
  return run(cwd, process.execPath, [path.join(root, checker)]);
}

function assertRejected(result, filenames) {
  assert.ifError(result.error);
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.match(result.stderr, /Generated website artifacts must not be tracked by Git/);
  for (const filename of filenames) assert.ok(result.stderr.includes(JSON.stringify(filename)), result.stderr);
  assert.match(result.stderr, /Remove these paths from the Git index/);
  assert.equal(result.stdout, '');
}

test('artifact guard retains every generated-root policy', () => {
  assert.deepEqual(generatedArtifactRoots, forbiddenRoots);
});

for (const directory of forbiddenRoots) {
  test(`rejects a clean tracked artifact in ${directory}`, async t => {
    const filename = directory + '/nested/old report.html';
    const root = await fixture(t, [filename]);
    assert.ok(git(root, 'ls-files', '-z').split('\0').includes(filename));
    assertClean(root); // The previous clean-tree-only gate accepts this fixture.
    assertRejected(check(root), [filename]);
    assertClean(root); // The guard is read-only, even when rejecting the fixture.
  });
}

test('allows real sources, canonical benchmark inputs, and similar directory names', async t => {
  const root = await fixture(t, [
    'frontend/src/index.ts', 'frontend/package-lock.json', 'package-lock.json',
    'results/benchmark_frontend_sources/source.json',
    'dev/tests/fixtures/benchmark_frontend/source.json',
    ...forbiddenRoots.map(directory => directory + '-source/fixture.txt'),
    'docs/source\nfrontend/node_modules/example.md',
  ]);
  const result = check(root, path.join(root, 'frontend'));
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /Verified no generated website artifacts are tracked/);
  assert.equal(result.stderr, '');
  assertClean(root);
});

test('allows ignored local artifacts that are not tracked', async t => {
  const root = await fixture(t);
  for (const directory of forbiddenRoots) await write(root, directory + '/generated.txt');
  assertClean(root);
  const result = check(root);
  assert.equal(result.status, 0, result.stdout + result.stderr);
});

test('rejects staged artifacts even when their working files are absent', async t => {
  const root = await fixture(t);
  const filename = 'frontend/test-results/staged-report.txt';
  await write(root, filename);
  git(root, 'add', '--force', '--', filename);
  assertRejected(check(root), [filename]);
  await rm(path.join(root, filename));
  assertRejected(check(root), [filename]);
});

test('reports newline, tab, quote, and Unicode filenames without Git quoting or line splitting', async t => {
  const filenames = ['frontend/playwright-report/report\nline\t"中文.html', 'node_modules/package name/index.js'];
  const root = await fixture(t, filenames);
  assertRejected(check(root), filenames);
  assertClean(root);
});

for (const kind of ['file', 'symlink']) {
  test(`rejects forbidden roots themselves when tracked as a ${kind}`, async t => {
    // Use non-overlapping roots so node_modules is not also a parent directory.
    const roots = ['node_modules', '.site-dist', 'frontend/node_modules'];
    const root = await fixture(t);
    for (const directory of roots) {
      await mkdir(path.dirname(path.join(root, directory)), { recursive: true });
      if (kind === 'symlink') await symlink('missing-generated-target', path.join(root, directory));
      else await write(root, directory);
    }
    git(root, 'add', '--force', '--all');
    await snapshot(root);
    assertRejected(check(root), roots);
    assertClean(root);
  });
}

test('Git enumeration failure fails the gate rather than reporting a clean checkout', async t => {
  const root = await fixture(t);
  await rm(path.join(root, '.git'), { recursive: true, force: true });
  const result = check(root);
  assert.equal(result.status, 1, result.stdout + result.stderr);
  assert.match(result.stderr, /Unable to inspect tracked website artifacts/);
  assert.match(result.stderr, /not a git repository/);
  assert.equal(result.stdout, '');
});

function eventSection(workflow, event) {
  const match = workflow.match(new RegExp(`^  ${event}:\\n([\\s\\S]*?)(?=^  [a-z_]+:|^\\S|$(?![\\s\\S]))`, 'm'));
  assert.ok(match, `${event} trigger must exist`);
  return match[0];
}

function matchesWorkflowPath(input, pattern) {
  // GitHub's path filters match dot-prefixed components too. Node's filesystem
  // glob defaults skip those components, so model this deliberately restricted
  // selector syntax using GitHub's documented * (no slash) / ** (any character).
  // https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#patterns-to-match-file-paths
  const expression = pattern.replaceAll('.', '\\.').replaceAll('**', '\0')
    .replaceAll('*', '[^/]*').replaceAll('\0', '[\\s\\S]*');
  return new RegExp('^' + expression + '$').test(input);
}

test('workflow selector model includes hidden paths and respects directory boundaries', () => {
  assert.ok(matchesWorkflowPath('docs/.vitepress/.temp/routes.js', 'docs/**'));
  assert.ok(matchesWorkflowPath('.site-dist/assets/a.js', '.site-dist/**'));
  assert.ok(matchesWorkflowPath('scripts/check-tracked-artifacts.mjs', 'scripts/**'));
  assert.equal(matchesWorkflowPath('.site-dist', '.site-dist/**'), false);
  assert.equal(matchesWorkflowPath('.site-dist-backup/a.js', '.site-dist/**'), false);
  assert.equal(matchesWorkflowPath('docs/a/b.md', 'docs/*.md'), false);
});

function assertTriggerCoverage(workflow) {
  for (const event of ['push', 'pull_request']) {
    const section = eventSection(workflow, event);
    const list = section.split(/^    paths:\n/m)[1]?.split(/^    \S/m)[0];
    assert.ok(list, `${event}.paths must exist`);
    const lines = list.trimEnd().split('\n').filter(line => line.trim() && !/^\s*#/.test(line));
    assert.ok(lines.every(line => /^      - '[a-zA-Z0-9_./*\-]+'$/.test(line)),
      'Update artifact trigger tests for the new selector syntax');
    const patterns = lines.map(line => line.slice(9, -1));
    for (const input of [workflowPath, '.gitignore', checker, tests,
      ...forbiddenRoots.flatMap(directory => [directory, directory + '/nested/artifact.txt'])]) {
      assert.ok(patterns.some(pattern => matchesWorkflowPath(input, pattern)),
        `${event}.paths must select artifact policy input ${input}`);
    }
  }
}

function siteSteps(workflow) {
  const site = workflow.split('\n  site:\n')[1]?.split('\n  deploy:\n')[0];
  assert.ok(site, 'The site job must exist');
  return site;
}

function step(site, name) {
  const heading = `      - name: ${name}\n`;
  const body = site.split(heading)[1]?.split(/^      - /m)[0];
  assert.ok(body, `The ${name} gate must exist`);
  assert.doesNotMatch(body, /^        (?:if|continue-on-error):/m, `${name} must not be optional`);
  const commands = body.split('        run: |\n')[1]?.trimEnd().split('\n');
  assert.ok(commands?.length && commands.every(line => line.startsWith('          ')), `${name} must run commands`);
  return { position: site.indexOf(heading), commands: commands.map(line => line.slice(10)) };
}

function assertGateWiring(workflow) {
  const site = siteSteps(workflow);
  const guard = step(site, guardName);
  assert.deepEqual(guard.commands, [`node --test ${tests}`, `node ${checker}`], 'Run the guard tests and actual tracked-file check');
  const install = site.indexOf('      - name: Install build dependencies\n');
  assert.ok(install > guard.position && site.indexOf('npm ci') > guard.position,
    'Reject tracked artifacts before npm ci can remove or replace them');
  const beforeQa = step(site, cleanName);
  const afterQa = step(site, afterQaName);
  assert.deepEqual(beforeQa.commands, cleanCommands, 'Preserve the pre-browser clean-tree gate');
  assert.deepEqual(afterQa.commands, cleanCommands, 'Check the complete tree again after browser QA');
  for (const command of ['npm run test:e2e --prefix frontend', 'npm run test:e2e:production --prefix frontend']) {
    const position = site.indexOf(`run: ${command}\n`);
    assert.ok(beforeQa.position < position && position < afterQa.position, `Keep clean-tree gates around ${command}`);
  }
  assert.ok(afterQa.position < site.indexOf('      - name: Upload GitHub Pages artifact\n'),
    'Validate browser side effects before publishing');
}

test('both workflow events select artifacts, policy, implementation, and tests, including root files', async () => {
  const workflow = await readFile(path.join(repositoryRoot, workflowPath), 'utf8');
  assertTriggerCoverage(workflow);
  for (const event of ['push', 'pull_request']) {
    const section = eventSection(workflow, event);
    for (const selector of ['.gitignore', 'node_modules', 'node_modules/**', '.site-dist', '.site-dist/**', 'scripts/**', 'frontend/**', 'docs/**']) {
      const broken = workflow.replace(section, section.replace(`      - '${selector}'\n`, ''));
      assert.throws(() => assertTriggerCoverage(broken), /must select artifact policy input/, `${event}: ${selector}`);
    }
  }
});

test('workflow executes the guard before installs and preserves both clean-tree gates', async () => {
  const workflow = await readFile(path.join(repositoryRoot, workflowPath), 'utf8');
  assertGateWiring(workflow);
  for (const command of [`node --test ${tests}`, `node ${checker}`]) {
    assert.throws(() => assertGateWiring(workflow.replace(`          ${command}\n`, '')), /guard tests|must run commands/);
  }
  for (const name of [guardName, cleanName, afterQaName]) {
    const heading = `      - name: ${name}\n`;
    assert.throws(() => assertGateWiring(workflow.replace(heading, heading + '        if: false\n')), /must not be optional/);
    assert.throws(() => assertGateWiring(workflow.replace(heading, heading + '        continue-on-error: true\n')), /must not be optional/);
  }
  const guardBlock = `      - name: ${guardName}\n        run: |\n          node --test ${tests}\n          node ${checker}\n`;
  const lateGuard = workflow.replace(guardBlock, '').replace('      - name: Typecheck dashboard\n', guardBlock + '      - name: Typecheck dashboard\n');
  assert.throws(() => assertGateWiring(lateGuard), /before npm ci/);
  assert.throws(() => assertGateWiring(workflow.replace(`      - name: ${cleanName}\n`, '      - name: Removed pre-browser check\n')), /gate must exist/);
  assert.throws(() => assertGateWiring(workflow.replace(`      - name: ${afterQaName}\n`, '      - name: Removed post-browser check\n')), /gate must exist/);
});

for (const scenario of ['clean', 'ignored outputs', 'modified source', 'untracked source', 'staged source']) {
  test(`both workflow clean-tree commands handle ${scenario}`, async t => {
    const workflow = await readFile(path.join(repositoryRoot, workflowPath), 'utf8');
    const site = siteSteps(workflow);
    const root = await fixture(t);
    if (scenario === 'ignored outputs') await write(root, '.site-dist/index.html');
    if (scenario === 'modified source') await write(root, 'README.md', 'changed\n');
    if (scenario === 'untracked source') await write(root, 'untracked.txt');
    if (scenario === 'staged source') {
      await write(root, 'staged.txt');
      git(root, 'add', '--', 'staged.txt');
    }
    for (const name of [cleanName, afterQaName]) {
      const commands = step(site, name).commands;
      const result = run(root, 'bash', ['-e', '-o', 'pipefail', '-c', commands.join('\n')]);
      assert.ifError(result.error);
      const expected = ['clean', 'ignored outputs'].includes(scenario) ? 0 : 1;
      assert.equal(result.status, expected, `${name}: ${result.stdout}${result.stderr}`);
    }
  });
}
