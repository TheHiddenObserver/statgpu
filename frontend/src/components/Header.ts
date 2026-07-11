import type { BenchmarkData, ParseReport } from '../schema';
import type { AppState } from '../state';
import { resetDownstreamFilters } from '../state';
import { h } from '../utils/dom';

export function renderHeader(
  data: BenchmarkData,
  parseReport: ParseReport | null,
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const header = h('div', { class: 'header' });

  const left = h('div');
  left.innerHTML =
    '<strong style="font-size:16px;">statgpu</strong> <span style="color:#888;">Benchmark Dashboard</span>';

  const right = h('div', {
    style: 'display:flex; align-items:center; gap:12px;',
  });

  // Hardware selector
  const hwLabel = h('span', {}, 'Environment: ');
  const hwSelect = h('select', { id: 'env-select' });
  for (const env of data.environments) {
    const opt = h('option', { value: env.env_id }, env.label);
    if (env.env_id === state.selectedEnvId) opt.setAttribute('selected', '');
    hwSelect.appendChild(opt);
  }
  hwSelect.addEventListener('change', () => {
    state.selectedEnvId = (hwSelect as HTMLSelectElement).value;
    // Reset downstream filters via centralized helper
    resetDownstreamFilters(state, {
      clearModel: true, clearVariant: true, clearPenalty: true,
      clearSolver: true, clearScale: true, clearBackend: true, clearExternal: true,
    });
    state.tableLimit = 200;
    onUpdate();
  });
  right.appendChild(hwLabel);
  right.appendChild(hwSelect);

  // Parse report info (passed in, not fetched)
  if (parseReport) {
    const info = h(
      'span',
      { style: 'color:#666; font-size:12px;' },
      `${parseReport.runs_generated} runs from ${parseReport.files_parsed}/${parseReport.files_seen} files`,
    );
    right.appendChild(info);
  }

  header.appendChild(left);
  header.appendChild(right);
  return header;
}
