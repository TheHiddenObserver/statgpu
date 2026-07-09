import { h } from '../utils/dom';

/** Wraps existing empty-state messages without changing copy in Step 2.
 *  Message improvements deferred to Step 3. */
export function emptyStateMessage(text: string): HTMLElement {
  return h(
    'div',
    {
      style:
        'padding:40px; text-align:center; color:#999; font-size:14px;',
    },
    text,
  );
}
