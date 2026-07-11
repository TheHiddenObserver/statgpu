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

  const title = h(
    'div',
    { style: 'font-weight:bold; margin-bottom:6px;' },
    'Categories',
  );
  sidebar.appendChild(title);

  // Search
  const search = h('input', {
    type: 'text',
    placeholder: 'Search...',
    style:
      'width:100%; padding:4px; margin-bottom:6px; border:1px solid #ccc; border-radius:4px;',
  }) as HTMLInputElement;
  sidebar.appendChild(search);

  // Category checkboxes
  const catContainer = h('div', { id: 'category-list' });
  const catRows: HTMLElement[] = [];
  for (const cat of data.categories) {
    const row = h('div', {
      style:
        'display:flex; align-items:center; gap:4px; padding:2px 0; cursor:pointer;',
      'data-cat-name': `${cat.name_zh} ${cat.name_en}`.toLowerCase(),
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
      resetDownstreamFilters(state, { clearModel: true, clearVariant: true, clearPenalty: true, clearSolver: true, clearScale: true });
      onUpdate();
    });
    const label = h(
      'label',
      { for: `cat-${cat.category_id}` },
      ` ${cat.name_zh}`,
    );
    row.appendChild(cb);
    row.appendChild(label);
    catContainer.appendChild(row);
    catRows.push(row);
  }
  sidebar.appendChild(catContainer);

  // Wire up search
  search.addEventListener('input', () => {
    const q = search.value.toLowerCase();
    for (const row of catRows) {
      const catName = row.getAttribute('data-cat-name') ?? '';
      row.style.display = !q || catName.includes(q) ? '' : 'none';
    }
  });

  // Select all / none — keep checkbox DOM in sync with state
  const catCheckboxes = new Map<string, HTMLInputElement>();
  for (const cat of data.categories) {
    const cb = catContainer.querySelector<HTMLInputElement>(`#cat-${cat.category_id}`);
    if (cb) catCheckboxes.set(cat.category_id, cb);
  }

  const btnRow = h('div', {
    style: 'display:flex; gap:4px; margin-top:6px;',
  });
  const allBtn = h('button', {}, 'All');
  allBtn.addEventListener('click', () => {
    for (const cat of data.categories)
      state.selectedCategoryIds.add(cat.category_id);
    for (const cb of catCheckboxes.values()) cb.checked = true;
    resetDownstreamFilters(state, { clearModel: true, clearVariant: true, clearPenalty: true, clearSolver: true, clearScale: true });
    onUpdate();
  });
  const noneBtn = h('button', {}, 'None');
  noneBtn.addEventListener('click', () => {
    state.selectedCategoryIds.clear();
    for (const cb of catCheckboxes.values()) cb.checked = false;
    resetDownstreamFilters(state, { clearModel: true, clearVariant: true, clearPenalty: true, clearSolver: true, clearScale: true });
    onUpdate();
  });
  btnRow.appendChild(allBtn);
  btnRow.appendChild(noneBtn);
  sidebar.appendChild(btnRow);

  return sidebar;
}
