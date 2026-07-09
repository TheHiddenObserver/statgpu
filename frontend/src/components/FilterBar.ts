import type { BenchmarkData, Run } from '../schema';
import type { AppState } from '../state';
import { h } from '../utils/dom';
import { filterRuns, getUniqueValues, getUniqueScaleKeys } from '../data';

export function renderFilterBar(
  allRuns: Run[],
  data: BenchmarkData,
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const bar = h('div', { class: 'filter-bar' });

  const filtered = filterRuns(allRuns, state);

  // Model selector
  const modelIds = getUniqueValues(filtered, 'model_id');
  if (modelIds.length > 0) {
    bar.appendChild(h('span', {}, 'Model:'));
    const sel = h('select', { style: 'padding:2px 6px;' });
    sel.appendChild(h('option', { value: '' }, 'All'));
    for (const m of modelIds) {
      const opt = h('option', { value: m }, m);
      if (m === state.selectedModelId) opt.setAttribute('selected', '');
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      state.selectedModelId = (sel as HTMLSelectElement).value || null;
      state.selectedPenalty = null;
      state.selectedSolver = null;
      onUpdate();
    });
    bar.appendChild(sel);
  }

  // Penalty selector (appears after model selected)
  if (state.selectedModelId) {
    const penalties = getUniqueValues(
      filtered.filter((r) => r.model_id === state.selectedModelId),
      'penalty',
    );
    bar.appendChild(h('span', {}, 'Penalty:'));
    const sel = h('select', { style: 'padding:2px 6px;' });
    sel.appendChild(h('option', { value: '' }, 'All'));
    for (const p of penalties) {
      const opt = h('option', { value: p }, p || 'none');
      if (p === state.selectedPenalty) opt.setAttribute('selected', '');
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      state.selectedPenalty = (sel as HTMLSelectElement).value || null;
      state.selectedSolver = null;
      onUpdate();
    });
    bar.appendChild(sel);
  }

  // Solver selector (appears after penalty selected)
  if (state.selectedPenalty) {
    const solvers = getUniqueValues(
      filtered.filter(
        (r) =>
          r.model_id === state.selectedModelId &&
          r.penalty === state.selectedPenalty,
      ),
      'solver',
    );
    bar.appendChild(h('span', {}, 'Solver:'));
    const sel = h('select', { style: 'padding:2px 6px;' });
    sel.appendChild(h('option', { value: '' }, 'All'));
    for (const s of solvers) {
      const opt = h('option', { value: s }, s);
      if (s === state.selectedSolver) opt.setAttribute('selected', '');
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      state.selectedSolver = (sel as HTMLSelectElement).value || null;
      onUpdate();
    });
    bar.appendChild(sel);
  }

  // Scale chips — derive from data filtered by everything EXCEPT scale
  const scaleOptionState: AppState = {
    ...state,
    selectedScaleKeys: new Set(),
  };
  const scaleOptionRuns = filterRuns(allRuns, scaleOptionState);
  const scaleKeys = getUniqueScaleKeys(scaleOptionRuns);
  if (scaleKeys.length > 0) {
    bar.appendChild(h('span', {}, 'Scale:'));
    for (const sk of scaleKeys.slice(0, 15)) {
      const chip = h(
        'span',
        {
          style: `display:inline-block; padding:2px 6px; margin:1px; border-radius:4px; cursor:pointer;
            font-size:11px; border:1px solid #ccc;
            ${state.selectedScaleKeys.has(sk) ? 'background:#1890ff; color:#fff; border-color:#1890ff;' : ''}`,
        },
        data.runs.find((r) => r.scale.scale_key === sk)?.scale.label ?? sk,
      );
      chip.addEventListener('click', () => {
        if (state.selectedScaleKeys.has(sk))
          state.selectedScaleKeys.delete(sk);
        else state.selectedScaleKeys.add(sk);
        onUpdate();
      });
      bar.appendChild(chip);
    }
  }

  // Backend radio
  bar.appendChild(h('span', {}, '| Backend:'));
  for (const bk of ['all', 'numpy', 'cupy', 'torch']) {
    const label = bk === 'all' ? 'All' : bk;
    const radio = h('label', {
      style: 'margin:0 4px; cursor:pointer; font-size:12px;',
    });
    const inp = h('input', {
      type: 'radio',
      name: 'backend',
      value: bk,
    }) as HTMLInputElement;
    if (bk === 'all' && state.selectedBackends.size === 0) inp.checked = true;
    if (bk !== 'all' && state.selectedBackends.has(bk)) inp.checked = true;
    inp.addEventListener('change', () => {
      if (bk === 'all') state.selectedBackends.clear();
      else {
        state.selectedBackends.clear();
        state.selectedBackends.add(bk);
      }
      onUpdate();
    });
    radio.appendChild(inp);
    radio.appendChild(document.createTextNode(label));
    bar.appendChild(radio);
  }

  // External frameworks
  bar.appendChild(h('span', {}, '| Ext:'));
  for (const ext of ['sklearn', 'glmnet', 'statsmodels']) {
    const lbl = h('label', {
      style: 'margin:0 4px; cursor:pointer; font-size:12px;',
    });
    const cb = h('input', {
      type: 'checkbox',
      value: ext,
    }) as HTMLInputElement;
    if (state.showExternal.has(ext)) cb.checked = true;
    cb.addEventListener('change', () => {
      if (cb.checked) state.showExternal.add(ext);
      else state.showExternal.delete(ext);
      onUpdate();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(ext));
    bar.appendChild(lbl);
  }

  return bar;
}
