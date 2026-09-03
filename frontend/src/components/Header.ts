import type { BenchmarkData, ParseReport, SourceInventory } from '../schema';
import type { AppState } from '../state';
import { setSelectedEnvironment } from '../state';
import { h } from '../utils/dom';

function renderBrand(): HTMLElement {
  const brand = h('div', { class: 'header-brand' });
  const homeLink = h(
    'a',
    {
      class: 'header-home-link',
      href: new URL('../', window.location.href).toString(),
      'aria-label': 'Back to statgpu home',
    },
    '← Home / 首页',
  );
  const title = h('span', { class: 'header-title' });
  title.appendChild(h('strong', { class: 'header-logo' }, 'statgpu'));
  title.appendChild(h('span', { class: 'header-subtitle' }, 'Benchmark Dashboard'));
  brand.appendChild(homeLink);
  brand.appendChild(title);
  return brand;
}

export function renderLoadingHeader(): HTMLElement {
  const header = h('div', { class: 'header header-loading' });
  header.appendChild(renderBrand());
  header.appendChild(
    h('span', { class: 'header-loading-status', role: 'status' }, 'Loading data and charts…'),
  );
  return header;
}

export function renderHeader(
  data: BenchmarkData,
  parseReport: ParseReport | null,
  sourceInventory: SourceInventory | null,
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const header = h('div', { class: 'header' });

  const brand = renderBrand();

  const controls = h('div', { class: 'header-controls' });

  // Hardware selector
  const hwLabel = h(
    'label',
    { for: 'env-select', class: 'header-env-label' },
    'Hardware environment:',
  );
  const hwSelect = h('select', { id: 'env-select' });
  for (const env of data.environments) {
    const sessionCount = env.member_env_ids?.length ?? 1;
    const opt = h(
      'option',
      {
        value: env.env_id,
        title: sessionCount > 1
          ? sessionCount + ' benchmark sessions on the same physical hardware'
          : 'One benchmark session',
      },
      env.label,
    );
    if (env.env_id === state.selectedEnvId) opt.setAttribute('selected', '');
    hwSelect.appendChild(opt);
  }
  hwSelect.addEventListener('change', () => {
    const selected = data.environments.find(
      env => env.env_id === (hwSelect as HTMLSelectElement).value,
    );
    if (!selected) return;
    setSelectedEnvironment(state, selected);
    onUpdate();
  });
  controls.appendChild(hwLabel);
  controls.appendChild(hwSelect);

  if (parseReport) {
    controls.appendChild(
      h(
        'span',
        { class: 'header-meta' },
        `${parseReport.runs_generated} runs from ${parseReport.files_parsed}/${parseReport.files_seen} registered files`,
      ),
    );
  }

  if (sourceInventory) {
    const inventoryText = [
      `${sourceInventory.registered_sources} registered`,
      `${sourceInventory.eligible_sources} eligible`,
      `${sourceInventory.not_canonical_ready_sources} non-ready`,
      `${sourceInventory.historical_or_excluded_sources} historical/excluded`,
    ].join(' · ');
    controls.appendChild(
      h(
        'span',
        {
          class: 'header-meta inventory-meta',
          title:
            `${sourceInventory.discovered_json_artifacts} discovered JSON artifacts; ` +
            `${sourceInventory.classified_candidate_sources} classified; ` +
            `${sourceInventory.eligible_unregistered_sources} eligible but unregistered; ` +
            `${sourceInventory.unclassified_artifacts} unclassified`,
        },
        inventoryText,
      ),
    );
  }

  header.appendChild(brand);
  header.appendChild(controls);
  return header;
}
