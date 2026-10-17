import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import {
  checkSourceDocuments,
  createSourceLinkChecker,
  sourceDocumentPaths,
} from './check-source-document-links.mjs';

const repositoryRoot = path.resolve('.');

test('all eleven source entry pages resolve files and GitHub fragments without site routing', async () => {
  const result = await checkSourceDocuments(repositoryRoot);
  assert.equal(result.pages, 11);
  assert.ok(sourceDocumentPaths.includes('frontend/README.md'));
  assert.ok(sourceDocumentPaths.includes('frontend/docs/benchmark-dashboard-maintenance.md'));
  assert.ok(result.checked >= 201, `Expected all source links, checked ${result.checked}`);
  assert.deepEqual(result.errors, []);
});

test('source checker rejects every former site-only route in each affected page', async () => {
  const checker = await createSourceLinkChecker(repositoryRoot);
  const broken = [
    '/dashboard/',
    '/dashboard/data/source_inventory.json',
    '/dashboard/data/parse_report.json',
    '/dashboard/data/benchmark_data.json',
    '/en/getting-started/quickstart',
    '/cn/',
  ];
  for (const page of sourceDocumentPaths) {
    const source = await readFile(page, 'utf8');
    for (const href of broken) {
      const result = await checker.checkDocument(page, source + `\n[regression](${href})\n`);
      assert.equal(result.errors.length, 1, `${page}: ${href}`);
      assert.match(result.errors[0], /missing repository target|generated file is not a source-reader target/);
    }
  }
});

test('both contributor pages reject all six former generated-data links, with or without generated files', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'statgpu-contributor-links-'));
  try {
    await mkdir(path.join(root, 'docs'), { recursive: true });
    const checker = await createSourceLinkChecker(root);
    const pages = [
      ['frontend/README.md', 'public/data/'],
      ['frontend/docs/benchmark-dashboard-maintenance.md', '../public/data/'],
    ];
    const files = ['source_inventory.json', 'parse_report.json', 'benchmark_data.json'];
    for (const generatedFilesExist of [false, true]) {
      if (generatedFilesExist) {
        await mkdir(path.join(root, 'frontend/public/data'), { recursive: true });
        for (const file of files) {
          await writeFile(path.join(root, 'frontend/public/data', file), '{}');
        }
      }
      for (const [page, prefix] of pages) {
        for (const file of files) {
          const href = prefix + file;
          const result = await checker.checkDocument(page, `[generated](${href})`);
          assert.equal(result.checked, 1);
          assert.deepEqual(result.errors, [
            `${page}: generated file is not a source-reader target: ${href}`,
          ], `${page}: ${href}; generated files exist: ${generatedFilesExist}`);
        }
      }
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('source checker verifies source files and anchors in a checkout with no generated data', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'statgpu-source-links-'));
  try {
    await mkdir(path.join(root, 'docs'), { recursive: true });
    await writeFile(path.join(root, 'docs/index.md'), '# Entry\n');
    await writeFile(path.join(root, 'docs/guide.md'), [
      '# First heading',
      '## First heading',
      '## Unicode 中文与标点：示例',
      '## A `code` and [link](https://example.org) heading!',
      '<a id="stable-anchor"></a>',
      '```md',
      '## Not a real heading',
      '<a id="not-an-anchor"></a>',
      '```',
    ].join('\n'));
    const checker = await createSourceLinkChecker(root);
    const good = [
      'guide.md#first-heading',
      'guide.md#first-heading-1',
      'guide.md#unicode-中文与标点示例',
      'guide.md#a-code-and-link-heading',
      'guide.md#stable-anchor',
      '#local-anchor',
    ].map(href => `[valid](${href})`).join('\n');
    const source = `---\nhero:\n  link: /en/site-only\n---\n# Local anchor\n${good}`;
    assert.deepEqual((await checker.checkDocument('docs/index.md', source)).errors, []);
    for (const href of [
      'missing.md', 'guide.md#missing-anchor', 'guide.md#not-a-real-heading',
      'guide.md#not-an-anchor', '../../outside.md', 'guide.md#bad%zz',
    ]) {
      assert.equal((await checker.checkDocument('docs/index.md', `[bad](${href})`)).errors.length, 1, href);
    }
    // Creating an ignored output must not make a GitHub/source link valid.
    await mkdir(path.join(root, 'frontend/public/data'), { recursive: true });
    await writeFile(path.join(root, 'frontend/public/data/benchmark_data.json'), '{}');
    const result = await checker.checkDocument('docs/index.md',
      '[generated](../frontend/public/data/benchmark_data.json)');
    assert.match(result.errors[0], /generated file is not a source-reader target/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('reference and HTML links are checked; fenced and inline code is not treated as navigation', async () => {
  const checker = await createSourceLinkChecker(repositoryRoot);
  const result = await checker.checkDocument('docs/index.md', [
    '[missing][reference]',
    '',
    '[reference]: en/missing.md',
    '',
    '<a href="en/missing.html">missing</a>',
    '`[not a link](missing-inline.md)`',
    '```md',
    '[not a link](missing-fenced.md)',
    '```',
  ].join('\n'));
  assert.equal(result.errors.length, 2);
  assert.ok(result.errors.some(error => error.includes('en/missing.md')));
  assert.ok(result.errors.some(error => error.includes('en/missing.html')));
});
