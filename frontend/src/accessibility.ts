/** Progressive accessibility hooks for generated dashboard controls. */

function activateWithKeyboard(element: HTMLElement): void {
  element.addEventListener('keydown', (event) => {
    const keyEvent = event as KeyboardEvent;
    if (keyEvent.key !== 'Enter' && keyEvent.key !== ' ') return;
    keyEvent.preventDefault();
    element.click();
  });
}

function labelFilterSelects(root: ParentNode): void {
  for (const select of root.querySelectorAll<HTMLSelectElement>('.filter-bar select')) {
    if (select.getAttribute('aria-label') || select.labels?.length) continue;
    const previous = select.previousElementSibling;
    const label = previous?.classList.contains('filter-label')
      ? previous.textContent?.replace(/:\s*$/, '').trim()
      : null;
    if (label) select.setAttribute('aria-label', label);
  }
}

function enhanceScaleChips(root: ParentNode): void {
  for (const chip of root.querySelectorAll<HTMLElement>('.scale-chip')) {
    chip.setAttribute('role', 'button');
    chip.setAttribute('tabindex', '0');
    if (!chip.getAttribute('aria-label')) {
      chip.setAttribute('aria-label', `Scale ${chip.textContent?.trim() ?? ''}`);
    }
    activateWithKeyboard(chip);
  }
}

function enhanceOverviewSorting(root: ParentNode): void {
  const table = root.querySelector('.overview-table-title + table');
  if (!table) return;
  for (const header of table.querySelectorAll<HTMLElement>('thead th')) {
    header.setAttribute('scope', 'col');
    header.setAttribute('tabindex', '0');
    const text = header.textContent ?? '';
    header.setAttribute(
      'aria-sort',
      text.includes('▲') ? 'ascending' : text.includes('▼') ? 'descending' : 'none',
    );
    header.setAttribute('aria-label', `Sort by ${text.replace(/[▲▼]/g, '').trim()}`);
    activateWithKeyboard(header);
  }
}

/** Apply keyboard and naming contracts after each generated render. */
export function enhanceDashboardAccessibility(root: ParentNode): void {
  labelFilterSelects(root);
  enhanceScaleChips(root);
  enhanceOverviewSorting(root);
}
