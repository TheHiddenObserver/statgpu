import path from 'node:path';
import { readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import GithubSlugger from 'github-slugger';
import { parseFragment } from 'parse5';
import { createMarkdownRenderer } from 'vitepress';

// These reader and contributor entry points must work from a source checkout.
// This check uses repository semantics, independently of site routing.
export const sourceDocumentPaths = Object.freeze([
  'docs/index.md',
  'docs/en/index.md',
  'docs/cn/index.md',
  'docs/cn/README.md',
  'docs/en/benchmarks.md',
  'docs/cn/benchmarks.md',
  'docs/en/guides/benchmarks.md',
  'docs/cn/guides/benchmarks.md',
  'docs/en/guides/statgpu_benchmark_dashboard.md',
  'frontend/README.md',
  'frontend/docs/benchmark-dashboard-maintenance.md',
]);

const external = /^(?:[a-z][a-z\d+.-]*:|\/\/)/i;
const generated = /^(?:\.site-dist(?:\/|$)|frontend\/(?:public\/data|dist)(?:\/|$)|docs\/assets\/benchmarks(?:\/|$)|dashboard(?:\/|$))/;

function body(source) {
  // YAML home-page configuration belongs to VitePress. The Markdown body is
  // also a GitHub entry point and must work without that configuration.
  return source.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
}

function visitHtml(html, visit) {
  function walk(node) {
    visit(node);
    for (const child of node.childNodes ?? []) walk(child);
  }
  walk(parseFragment(html));
}

export async function createSourceLinkChecker(repositoryRoot) {
  const renderer = await createMarkdownRenderer(path.join(repositoryRoot, 'docs'), {
    highlight: code => code,
  });
  // Use VitePress's MarkdownIt implementation with no VitePress plugins. A new
  // parser prevents website rewriting, heading permalinks, or its singleton
  // renderer cache from turning broken repository links into accepted routes.
  const md = new renderer.constructor({ html: true });
  const anchorCache = new Map();

  function inspect(source) {
    const tokens = md.parse(body(source), {});
    const links = [];
    const anchors = new Set();
    const slugger = new GithubSlugger();
    for (let i = 0; i < tokens.length; i += 1) {
      const token = tokens[i];
      if (token.type === 'heading_open') {
        const heading = tokens[i + 1].children ?? [];
        const text = heading.filter(child => ['text', 'code_inline', 'image'].includes(child.type))
          .map(child => child.content).join('');
        anchors.add(slugger.slug(text));
      }
      for (const child of [token, ...(token.children ?? [])]) {
        if (child.type === 'link_open') links.push(child.attrGet('href'));
        if (child.type === 'image') links.push(child.attrGet('src'));
        if (child.type === 'html_block' || child.type === 'html_inline') {
          visitHtml(child.content, node => {
            for (const attribute of node.attrs ?? []) {
              if (attribute.name === 'id' || (node.tagName === 'a' && attribute.name === 'name')) {
                anchors.add(attribute.value);
              }
              if (attribute.name === 'href' || attribute.name === 'src') links.push(attribute.value);
            }
          });
        }
      }
    }
    return { links, anchors };
  }

  async function checkDocument(relativePath, source) {
    const fullPath = path.resolve(repositoryRoot, relativePath);
    source ??= await readFile(fullPath, 'utf8');
    const { links, anchors } = inspect(source);
    const errors = [];
    let checked = 0;
    for (const href of links) {
      if (!href || external.test(href)) continue;
      checked += 1;
      let pathname;
      let fragment;
      try {
        const [target, hash] = href.split('#', 2);
        pathname = decodeURIComponent(target.split('?', 1)[0]);
        fragment = hash ? decodeURIComponent(hash) : '';
      } catch {
        errors.push(`${relativePath}: invalid URL encoding: ${href}`);
        continue;
      }
      const target = !pathname ? fullPath : pathname.startsWith('/')
        ? path.resolve(repositoryRoot, '.' + pathname)
        : path.resolve(path.dirname(fullPath), pathname);
      const repoRelative = path.relative(repositoryRoot, target).replace(/\\/g, '/');
      const fail = reason => errors.push(`${relativePath}: ${reason}: ${href}`);
      if (repoRelative === '..' || repoRelative.startsWith('../') || path.isAbsolute(repoRelative)) {
        fail('link escapes repository');
        continue;
      }
      if (generated.test(repoRelative)) {
        fail('generated file is not a source-reader target');
        continue;
      }
      let info;
      try { info = await stat(target); } catch { /* Report missing below. */ }
      if (!info) {
        fail('missing repository target');
        continue;
      }
      if (fragment && /\.(?:md|markdown)$/i.test(target)) {
        let targetAnchors = target === fullPath ? anchors : anchorCache.get(target);
        if (!targetAnchors) {
          targetAnchors = inspect(await readFile(target, 'utf8')).anchors;
          anchorCache.set(target, targetAnchors);
        }
        if (!targetAnchors.has(fragment)) fail('missing repository anchor');
      }
    }
    return { relativePath, checked, errors };
  }

  return { checkDocument, inspect };
}

export async function checkSourceDocuments(repositoryRoot, paths = sourceDocumentPaths) {
  const checker = await createSourceLinkChecker(repositoryRoot);
  const results = [];
  for (const relativePath of paths) results.push(await checker.checkDocument(relativePath));
  return {
    pages: results.length,
    checked: results.reduce((sum, result) => sum + result.checked, 0),
    errors: results.flatMap(result => result.errors),
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const result = await checkSourceDocuments(root);
  if (result.errors.length) {
    console.error(result.errors.join('\n'));
    process.exitCode = 1;
  } else {
    console.log(`Verified ${result.checked} repository links and fragments in ${result.pages} source entry pages`);
  }
}
