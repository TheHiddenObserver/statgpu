import path from 'node:path';
import { readFile } from 'node:fs/promises';
import parse5 from 'parse5';

// Parse HTML rather than matching source text: entities, SVG attributes,
// unquoted attributes and literal markup inside scripts/comments all matter.
export function inspectHtml(html) {
  const ids = new Set();
  const duplicateIds = new Set();
  const references = [];
  const mathErrors = [];
  const walk = node => {
    const attrs = new Map((node.attrs ?? []).map(attr => [attr.name, attr.value]));
    const id = attrs.get('id');
    if (id !== undefined) {
      if (ids.has(id)) duplicateIds.add(id);
      ids.add(id);
    }
    if (node.tagName === 'a' && attrs.has('name')) ids.add(attrs.get('name'));
    for (const attribute of ['href', 'src']) {
      if (attrs.has(attribute)) references.push({ attribute, target: attrs.get(attribute) });
    }
    if (attrs.has('data-mjx-error') || attrs.get('data-mml-node') === 'merror'
        || node.tagName === 'merror' || node.tagName === 'mjx-merror') {
      mathErrors.push(attrs.get('data-mjx-error') ?? 'MathJax merror node');
    }
    // Template content is not in the live DOM and must not satisfy a link.
    for (const child of node.childNodes ?? []) walk(child);
  };
  walk(parse5.parseFragment(html));
  return { ids, duplicateIds, references, mathErrors };
}

export function resolveSiteTarget(htmlFile, target, { siteDir, siteBase, fileIndex }) {
  const raw = target.trim();
  if (!raw || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(raw)) return null;
  const pagePath = path.relative(siteDir, htmlFile).split(path.sep).join('/');
  const url = new URL(raw, `https://artifact.invalid${siteBase}${pagePath}`);
  if (url.origin !== 'https://artifact.invalid') {
    throw new Error(`Local URL escapes the deployment artifact: ${target}`);
  }
  const pathname = decodeURIComponent(url.pathname);
  if (!pathname.startsWith(siteBase)) {
    throw new Error(`Root-absolute URL is outside STATGPU_SITE_BASE: ${target}`);
  }
  const candidate = path.resolve(siteDir, pathname.slice(siteBase.length));
  const relative = path.relative(siteDir, candidate);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Local URL escapes the deployment artifact: ${target}`);
  }
  const file = [candidate, `${candidate}.html`, path.join(candidate, 'index.html')]
    .find(name => fileIndex.has(name));
  return { file, fragment: decodeURIComponent(url.hash.slice(1)) };
}

export async function validateSiteHtml({ siteDir, siteBase, allFiles }) {
  const fileIndex = new Set(allFiles.map(file => path.resolve(file)));
  const htmlFiles = allFiles.filter(file => file.endsWith('.html'));
  const pages = new Map(await Promise.all(htmlFiles.map(async file =>
    [path.resolve(file), inspectHtml(await readFile(file, 'utf8'))],
  )));
  const failures = [];
  for (const [file, page] of pages) {
    const label = path.relative(siteDir, file);
    for (const id of page.duplicateIds) failures.push(`${label}: duplicate HTML id ${JSON.stringify(id)}`);
    for (const error of page.mathErrors) failures.push(`${label}: MathJax error: ${error}`);
    for (const { attribute, target } of page.references) {
      if (target.includes('docs/assets/benchmarks')) {
        failures.push(`${label} -> ${target}: legacy dashboard path`);
        continue;
      }
      try {
        const resolved = resolveSiteTarget(file, target, { siteDir, siteBase, fileIndex });
        if (!resolved) continue;
        if (!resolved.file) {
          failures.push(`${label} -> ${target}: missing deployment file`);
          continue;
        }
        if (attribute === 'href' && resolved.fragment) {
          const targetPage = pages.get(resolved.file);
          // Non-HTML fragments (for example PDF page selectors) use the
          // destination format's own fragment semantics.
          if (targetPage && !targetPage.ids.has(resolved.fragment)) {
            failures.push(`${label} -> ${target}: missing HTML fragment ${JSON.stringify(resolved.fragment)}`);
          }
        }
      } catch (error) {
        failures.push(`${label} -> ${target}: ${error.message}`);
      }
    }
  }
  return failures;
}
