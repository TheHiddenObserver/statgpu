import type { BenchmarkData } from '../schema';
import type { AppState } from '../state';
import { resetDownstreamFilters } from '../state';
import { h } from '../utils/dom';

export function renderSidebar(
  data: BenchmarkData,
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const sidebar = h('div', { class: 'sidebar' });

  sidebar.appendChild(h('div', { class: 'sidebar-title' }, 'Categories'));

  const search = h('input', {
    type: 'text',
    placeholder: 'Search categories…',
    class: 'sidebar-search',
    'aria-label': 'Search categories',
  }) as HTMLInputElement;
  sidebar.appendChild(search);

  const catContainer = h('div', { id: 'category-list', class: 'category-list' });
  const catRows: HTMLElement[] = [];
  for (const cat of data.categories) {
    const row = h('div', {
      class: 'category-row',
      'data-cat-name': `${cat.name_en} ${cat.name_zh}`.toLowerCase(),
    });
    const cb = h('input', {
      type: 'checkbox',
      id: `cat-${cat.category_id}`,
      value: cat.category_id,
    }) as HTMLInputElement;
    if (state.selectedCategoryIds.has(cat.category_id)) cb.checked = true;
    cb.addEventListener('change', () => {
      if (cb.checked) state.selectedCategoryIds.add(cat.category_id);
      else state.selectedCategoryIds.delete(cat.category_id);
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
      onUpdate();
    });
    const label = h(
      'label',
      {
        for: `cat-${cat.category_id}`,
        title: cat.name_zh !== cat.name_en ? cat.name_zh : cat.name_en,
      },
      cat.name_en,
    );
    row.appendChild(cb);
    row.appendChild(label);
    catContainer.appendChild(row);
    catRows.push(row);
  }
  sidebar.appendChild(catContainer);

  search.addEventListener('input', () => {
    const query = search.value.toLowerCase();
    for (const row of catRows) {
      const categoryName = row.getAttribute('data-cat-name') ?? '';
      row.style.display = !query || categoryName.includes(query) ? '' : 'none';
    }
  });

  const catCheckboxes = new Map<string, HTMLInputElement>();
  for (const cat of data.categories) {
    const cb = catContainer.querySelector<HTMLInputElement>(`#cat-${cat.category_id}`);
    if (cb) catCheckboxes.set(cat.category_id, cb);
  }

  const btnRow = h('div', { class: 'sidebar-actions' });
  const allBtn = h('button', { type: 'button' }, 'All');
  allBtn.addEventListener('click', () => {
    for (const cat of data.categories) state.selectedCategoryIds.add(cat.category_id);
    for (const cb of catCheckboxes.values()) cb.checked = true;
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
    onUpdate();
  });
  const noneBtn = h('button', { type: 'button' }, 'None');
  noneBtn.addEventListener('click', () => {
    state.selectedCategoryIds.clear();
    for (const cb of catCheckboxes.values()) cb.checked = false;
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
    onUpdate();
  });
  btnRow.appendChild(allBtn);
  btnRow.appendChild(noneBtn);
  sidebar.appendChild(btnRow);

  return sidebar;
}
