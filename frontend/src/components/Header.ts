import type { BenchmarkData, ParseReport, SourceInventory } from '../schema';
import type { AppState } from '../state';
import { resetDownstreamFilters } from '../state';
import { h } from '../utils/dom';

export function renderHeader(
  data: BenchmarkData,
  parseReport: ParseReport | null,
  sourceInventory: SourceInventory | null,
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const header = h('div', { class: 'header' });

  const brand = h('div', { class: 'header-brand' });
  brand.appendChild(h('strong', { class: 'header-logo' }, 'statgpu'));
  brand.appendChild(h('span', { class: 'header-subtitle' }, 'Benchmark Dashboard'));

  const controls = h('div', { class: 'header-controls' });

  // Hardware selector
  const hwLabel = h('label', { for: 'env-select', class: 'header-env-label' }, 'Environment:');
  const hwSelect = h('select', { id: 'env-select' });
  for (const env of data.environments) {
    const opt = h('option', { value: env.env_id }, env.label);
    if (env.env_id === state.selectedEnvId) opt.setAttribute('selected', '');
    hwSelect.appendChild(opt);
  }
  hwSelect.addEventListener('change', () => {
    state.selectedEnvId = (hwSelect as HTMLSelectElement).value;
    resetDownstreamFilters(state, {
      clearMetricScope: true,
      clearModel: true,
      clearVariant: true,
      clearPenalty: true,
      clearSolver: true,
      clearScale: true,
      clearBackend: true,
      clearExternal: true,
    });
    state.tableLimit = 200;
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
