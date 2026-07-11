import type { BenchmarkData, Run } from '../schema';
import type { AppState } from '../state';
import {
  setSelectedModel,
  setSelectedVariant,
  setSelectedPenalty,
  setSelectedSolver,
  toggleScaleKey,
  setBackend,
  toggleExternal,
} from '../state';
import { h } from '../utils/dom';
import {
  filterRuns,
  getUniqueValues,
  getUniqueScaleKeys,
  getScaleLabelMap,
} from '../data';

export function renderFilterBar(
  allRuns: Run[],
  data: BenchmarkData,
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const bar = h('div', { class: 'filter-bar' });

  // Option runs: exclude self + downstream filters so selecting a value
  // doesn't shrink the dropdown to only that value (avoids stale-filter deadlock).
  const modelOptionState: AppState = {
    ...state,
    selectedModelId: null,
    selectedPenalty: null,
    selectedSolver: null,
    selectedScaleKeys: new Set(),
  };
  const modelOptionRuns = filterRuns(allRuns, modelOptionState);

  const penaltyOptionState: AppState = {
    ...state,
    selectedPenalty: null,
    selectedSolver: null,
    selectedScaleKeys: new Set(),
  };
  const penaltyOptionRuns = filterRuns(allRuns, penaltyOptionState);

  const solverOptionState: AppState = {
    ...state,
    selectedSolver: null,
  };
  const solverOptionRuns = filterRuns(allRuns, solverOptionState);

  // Model selector
  const modelIds = getUniqueValues(modelOptionRuns, 'model_id');
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
      setSelectedModel(state, (sel as HTMLSelectElement).value || null);
      onUpdate();
    });
    bar.appendChild(sel);
  }

  // Variant selector (appears after model selected, when variants exist)
  if (state.selectedModelId) {
    const variantRuns = filterRuns(allRuns, {
      ...state, selectedVariant: null, selectedPenalty: null, selectedSolver: null, selectedScaleKeys: new Set(),
    } as AppState);
    const variants = getUniqueValues(
      variantRuns.filter((r) => r.model_id === state.selectedModelId && r.variant),
      'variant',
    );
    if (variants.length > 0) {
      bar.appendChild(h('span', {}, 'Variant:'));
      const vsel = h('select', { style: 'padding:2px 6px;' });
      vsel.appendChild(h('option', { value: '' }, 'All'));
      for (const v of variants) {
        const opt = h('option', { value: v }, v);
        if (v === state.selectedVariant) opt.setAttribute('selected', '');
        vsel.appendChild(opt);
      }
      vsel.addEventListener('change', () => {
        const val = (vsel as HTMLSelectElement).value;
        setSelectedVariant(state, val || null);
        onUpdate();
      });
      bar.appendChild(vsel);
    }
  }

  // Penalty selector (appears after model selected)
  if (state.selectedModelId) {
    const penalties = getUniqueValues(
      penaltyOptionRuns.filter((r) => r.model_id === state.selectedModelId),
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
      setSelectedPenalty(state, (sel as HTMLSelectElement).value || null);
      onUpdate();
    });
    bar.appendChild(sel);
  }

  // Solver selector (appears after penalty selected)
  if (state.selectedPenalty) {
    const solvers = getUniqueValues(
      solverOptionRuns.filter(
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
      setSelectedSolver(state, (sel as HTMLSelectElement).value || null);
      onUpdate();
    });
    bar.appendChild(sel);
  }

  // Scale chips — derive from data filtered by everything EXCEPT scale
  const scaleOptionRuns = filterRuns(allRuns, state, { ignoreScale: true });
  const scaleKeys = getUniqueScaleKeys(scaleOptionRuns);
  if (scaleKeys.length > 0) {
    bar.appendChild(h('span', {}, 'Scale:'));
    const labelMap = getScaleLabelMap(data.runs);
    for (const sk of scaleKeys.slice(0, 15)) {
      const active = state.selectedScaleKeys.has(sk);
      const chip = h(
        'span',
        {
          class: `scale-chip${active ? ' active' : ''}`,
          'data-scale-key': sk,
          'aria-pressed': String(active),
        },
        labelMap.get(sk) ?? sk,
      );
      chip.addEventListener('click', () => {
        toggleScaleKey(state, sk);
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
      setBackend(
        state,
        bk === 'all' ? null : (bk as 'numpy' | 'cupy' | 'torch'),
      );
      onUpdate();
    });
    radio.appendChild(inp);
    radio.appendChild(document.createTextNode(label));
    bar.appendChild(radio);
  }

  // External frameworks (context-aware: only those with runs in current filter context)
  const extAvailableRuns = filterRuns(allRuns, state, { ignoreExternal: true });
  const extAvailable = new Set(extAvailableRuns.filter(r => r.framework !== 'statgpu').map(r => r.framework));
  const extFrameworks = data.frameworks.filter(f => f.external && extAvailable.has(f.framework_id));
  if (extFrameworks.length > 0) bar.appendChild(h('span', {}, '| Ext:'));
  for (const fw of extFrameworks) {
    const ext = fw.framework_id;
    const lbl = h('label', {
      style: 'margin:0 4px; cursor:pointer; font-size:12px;',
    });
    const cb = h('input', {
      type: 'checkbox',
      value: ext,
    }) as HTMLInputElement;
    if (state.showExternal.has(ext)) cb.checked = true;
    cb.addEventListener('change', () => {
      toggleExternal(state, ext);
      onUpdate();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(ext));
    bar.appendChild(lbl);
  }

  return bar;
}
