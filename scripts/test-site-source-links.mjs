import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { createMarkdownRenderer } from 'vitepress';
import config from '../docs/.vitepress/config.ts';

const docsRoot = path.resolve('docs');
const md = await createMarkdownRenderer(docsRoot, config.markdown, config.base);
const page = 'en/models/regression/lasso.md';
function render(source, relativePath = page) {
  return md.render(source, {
    relativePath,
    path: path.join(docsRoot, relativePath),
    cleanUrls: true,
  });
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

test('dashboard entry bypasses docs routing and retains configured base', () => {
  const html = render('[dashboard](/dashboard/)');
  assert.ok(html.includes(`href="${config.base}dashboard/"`));
  assert.match(html, /target="_self"/);
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
