import path from 'node:path';
import { existsSync, readFileSync } from 'node:fs';

const repositoryUrl = 'https://github.com/TheHiddenObserver/statgpu/blob/master/';
const dashboardGuide = 'en/guides/statgpu_benchmark_dashboard.md';

// Every generated resource has a real source-readable fallback. Keep this map
// exact: ordinary links to the reader guide must continue to open that guide.
export const generatedSourceRoutes = Object.freeze({
  'open-dashboard': 'dashboard/',
  'source-inventory': 'dashboard/data/source_inventory.json',
  'parse-report': 'dashboard/data/parse_report.json',
  'normalized-results': 'dashboard/data/benchmark_data.json',
});

function outside(relative) {
  return relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative);
}

export function installSourceLinks(md, { docsRoot, base }) {
  const repositoryRoot = path.dirname(docsRoot);
  md.core.ruler.after('inline', 'statgpu-source-links', state => {
    const relativePath = state.env.relativePath;
    if (!relativePath) return;
    const source = path.resolve(docsRoot, relativePath);
    for (const token of state.tokens) {
      for (const child of token.children ?? []) {
        if (child.type !== 'link_open') continue;
        const href = child.attrGet('href');
        if (!href || /^(?:[a-z]+:|\/\/)/i.test(href)) continue;
        // Retain compatibility for site-only configuration links; maintained
        // Markdown uses the source fallbacks rather than these root URLs.
        if (href === '/dashboard/' || href === '/dashboard') {
          child.attrSet('href', base + 'dashboard/');
          child.attrSet('target', '_self');
          continue;
        }
        const [targetWithQuery, fragment] = href.split('#', 2);
        const [target, query] = targetWithQuery.split('?', 2);
        const absolute = target ? path.resolve(path.dirname(source), decodeURI(target)) : source;
        const repoRelative = path.relative(repositoryRoot, absolute).replace(/\\/g, '/');
        const docsRelative = path.relative(docsRoot, absolute);
        const generatedRoute = docsRelative.replace(/\\/g, '/') === dashboardGuide
          && Object.hasOwn(generatedSourceRoutes, fragment) && generatedSourceRoutes[fragment];
        if (generatedRoute) {
          // A renamed/deleted fallback must fail even though its generated
          // counterpart could still exist in a prior build directory.
          if (!existsSync(absolute) || !readFileSync(absolute, 'utf8').includes(`<a id="${fragment}"></a>`)) {
            throw new Error(`Invalid generated-resource source fallback in ${relativePath}: ${href}`);
          }
          child.attrSet('href', base + generatedRoute + (query ? '?' + query : ''));
          // Skip VitePress's Markdown routing and avoid a second base prefix.
          child.attrSet('target', '_self');
          continue;
        }
        if (href.startsWith('#')) continue;
        if (!href.startsWith('/') && (outside(docsRelative) || target.endsWith('.markdown'))) {
          if (outside(path.relative(repositoryRoot, absolute)) || !existsSync(absolute)) {
            throw new Error(`Invalid repository link in ${relativePath}: ${href}`);
          }
          child.attrSet('href', repositoryUrl + repoRelative
            + (query ? '?' + query : '') + (fragment ? '#' + fragment : ''));
        }
      }
    }
  });
}
