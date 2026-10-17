import GithubSlugger from 'github-slugger';
import { inspectHtml } from '../../scripts/site-html-validation.mjs';

// Keep the previously published VitePress 1.6.4 spelling as an alias when it
// does not conflict with a source-compatible heading or an explicit HTML ID.
function vitepressSlug(title) {
  return title.normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[\u0000-\u001f]/g, '')
    .replace(/[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g, '-')
    .replace(/-{2,}/g, '-').replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1').toLowerCase();
}

function uniqueId(id, used) {
  let candidate = id;
  for (let suffix = 1; used.has(candidate); suffix++) candidate = `${id}-${suffix}`;
  used.add(candidate);
  return candidate;
}

export function installHeadingAnchors(md) {
  // Run after inline/attribute processing, but before VitePress builds its
  // permalinks. All headings are assigned together so an early alias cannot
  // steal a later heading's canonical identifier.
  md.core.ruler.before('anchor', 'statgpu-heading-ids', state => {
    const headings = [];
    const used = new Set();
    const legacyIds = new Set();
    for (const token of state.tokens) {
      for (const child of [token, ...(token.children ?? [])]) {
        if (child.type === 'html_block' || child.type === 'html_inline') {
          for (const id of inspectHtml(child.content).ids) used.add(id);
        }
      }
      const id = token.attrGet('id');
      if (id !== null) used.add(id);
    }
    const slugger = new GithubSlugger();
    for (let index = 0; index < state.tokens.length; index++) {
      const token = state.tokens[index];
      if (token.type !== 'heading_open') continue;
      const inline = state.tokens[index + 1];
      const title = (inline.children ?? [])
        .filter(child => ['text', 'code_inline'].includes(child.type))
        .map(child => child.content).join('');
      const explicit = token.attrGet('id');
      const legacy = uniqueId(explicit ?? vitepressSlug(title), legacyIds);
      let id = explicit;
      if (id === null) {
        do { id = slugger.slug(title); } while (used.has(id));
        used.add(id);
        token.attrSet('id', id);
      }
      headings.push({ token, inline, legacy, id });
    }
    for (const { token, legacy, id } of headings) {
      if (legacy !== id && !used.has(legacy)) {
        used.add(legacy);
        token.meta = { ...token.meta, statgpuLegacyId: legacy };
      }
    }
  });
  const renderHeading = md.renderer.rules.heading_open
    ?? ((tokens, index, options, env, self) => self.renderToken(tokens, index, options));
  md.renderer.rules.heading_open = (tokens, index, options, env, self) => {
    const opening = renderHeading(tokens, index, options, env, self);
    const alias = tokens[index].meta?.statgpuLegacyId;
    return opening + (alias === undefined ? ''
      : `<span id="${md.utils.escapeHtml(alias)}" class="ignore-header" aria-hidden="true"></span>`);
  };
}
