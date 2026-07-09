import { h } from '../utils/dom';

/** Improved empty state messages — Step 3 UI polish. */
export function emptyStateMessage(text: string): HTMLElement {
  return h('div', { class: 'empty-state' }, text);
}

export function emptyFilterMessage(): HTMLElement {
  return h(
    'div',
    { class: 'empty-state' },
    'No runs match the current filters.',
    h('br'),
    h(
      'small',
      {},
      'Try clearing scale, solver, or external framework filters.',
    ),
  );
}

export function emptyChartMessage(): HTMLElement {
  return h(
    'div',
    { class: 'empty-state' },
    'No timing data is available for this filtered view.',
  );
}
