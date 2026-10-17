import DefaultTheme from 'vitepress/theme';
import localSearchIndex from '@localSearchIndex';
import './custom.css';

type SearchIndexModule = { default: string };
type SearchIndexLoaders = Record<
  string,
  (() => Promise<SearchIndexModule>) | undefined
>;

const searchIndexLoaders = localSearchIndex as SearchIndexLoaders;
const prefetchedLocales = new Set<string>();

function prefetchCurrentSearchIndex() {
  const locale = document.documentElement.lang.toLowerCase().startsWith('zh')
    ? 'cn'
    : 'root';
  if (prefetchedLocales.has(locale)) return;

  const load = searchIndexLoaders[locale];
  if (!load) return;
  prefetchedLocales.add(locale);
  void load().catch(() => prefetchedLocales.delete(locale));
}

function installSearchIndexPrefetch() {
  // Start immediately when the user shows search intent. The asynchronous
  // startup warm-up runs after hydration begins, so a normal first search
  // does not wait for the browser's sometimes heavily delayed idle callback.
  document.addEventListener(
    'pointerover',
    event => {
      if ((event.target as Element | null)?.closest('.VPNavBarSearch')) {
        prefetchCurrentSearchIndex();
      }
    },
    { passive: true },
  );
  window.addEventListener('keydown', event => {
    if (
      (event.key.toLowerCase() === 'k' && (event.ctrlKey || event.metaKey)) ||
      event.key === '/'
    ) {
      prefetchCurrentSearchIndex();
    }
  });

  window.setTimeout(prefetchCurrentSearchIndex, 0);
}

export default {
  extends: DefaultTheme,
  enhanceApp() {
    if (typeof window !== 'undefined') installSearchIndexPrefetch();
  },
};
