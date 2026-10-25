import { h, clear } from '../utils/dom';
import { parseDiagnosticPreview, previewDuration, previewScale, previewValue, previewOperation, type DiagnosticPreview } from '../preview';

function detail(label: string, value: unknown): HTMLElement {
  const details = h('details', { class: 'preview-detail' });
  details.appendChild(h('summary', {}, label));
  details.appendChild(h('pre', {}, value == null ? 'Not recorded' : JSON.stringify(value, null, 2)));
  return details;
}
function table(headers: string[], rows: Array<Array<string | HTMLElement>>): HTMLElement {
  const wrap = h('div', { class: 'preview-table-wrap' });
  const element = h('table');
  const head = h('thead');
  const headings = h('tr');
  headers.forEach(label => headings.appendChild(h('th', { scope: 'col' }, label)));
  head.appendChild(headings);
  element.appendChild(head);
  const body = h('tbody');
  for (const cells of rows) {
    const row = h('tr');
    cells.forEach(cell => row.appendChild(h('td', {}, cell)));
    body.appendChild(row);
  }
  element.appendChild(body);
  wrap.appendChild(element);
  return wrap;
}
function renderLoaded(preview: DiagnosticPreview, filename: string): HTMLElement {
  const root = h('div', { class: 'preview-loaded' });
  root.appendChild(h('h2', {}, `Diagnostic preview: ${filename}`));
  root.appendChild(h('p', { class: 'preview-warning' }, 'Diagnostic only. No canonical performance promotion or speedup comparisons. This browser displays stored statuses; it does not authenticate hashes or rerun validators. Correctness reassessment and operator attestation do not establish machine-verified GPU exclusivity.'));
  root.appendChild(h('p', {}, `${preview.campaigns.length} campaigns · ${preview.jobs.length} planned jobs · ${preview.records.length} recorded timing fields`));
  root.appendChild(h('p', {}, `Machine-verified exclusivity: no (${preview.exclusivity.machine_status}). Operator attestation: ${preview.exclusivity.operator_attestation == null ? 'not recorded' : 'recorded, not machine verification'}.`));
  root.appendChild(detail('Operator attestation', preview.exclusivity.operator_attestation));
  root.appendChild(detail('Campaign provenance and coverage', { campaigns: preview.campaigns, coverage_scope: preview.coverage_scope, qualification_scope: preview.qualification_scope, validator_source: preview.validator_source, coverage_ledger: preview.coverage_ledger }));
  root.appendChild(h('h3', {}, 'Planned jobs, including blocked and not run'));
  root.appendChild(table(['Suite / lane', 'Reported status', 'Recorded validation (unverified here)', 'Recorded correctness', 'Evidence'], preview.jobs.map(job => [job.suite_or_lane, previewValue(job.reported_status), job.validation_status, job.correctness_status, detail('Full job evidence', job)])));
  const controls = h('div', { class: 'preview-controls' });
  controls.appendChild(h('label', { for: 'preview-job-filter' }, 'Timing job:'));
  const select = h('select', { id: 'preview-job-filter' }) as HTMLSelectElement;
  select.appendChild(h('option', { value: '' }, 'All planned jobs'));
  preview.jobs.forEach(job => select.appendChild(h('option', { value: job.job_id }, `${job.suite_or_lane} · ${job.job_id}`)));
  controls.appendChild(select);
  controls.appendChild(h('label', { for: 'preview-search' }, 'Search timings:'));
  const search = h('input', { id: 'preview-search', type: 'search', placeholder: 'Scope, operation, backend…' }) as HTMLInputElement;
  controls.appendChild(search);
  root.appendChild(controls);
  const records = h('div', { id: 'preview-records' });
  let limit = 200;
  const renderRecords = (): void => {
    clear(records);
    const query = search.value.toLowerCase().trim();
    const filtered = preview.records.filter(record => (!select.value || record.job_id === select.value) && (!query || JSON.stringify(record).toLowerCase().includes(query)));
    records.appendChild(h('h3', {}, `Recorded timing fields (${Math.min(filtered.length, limit)} of ${filtered.length})`));
    if (filtered.length === 0) { records.appendChild(h('p', {}, 'No recorded timing fields. Jobs without timing remain visible above.')); return; }
    records.appendChild(table(['Suite / lane', 'Operation / backend', 'Scale', 'Exact timing scope', 'Recorded summary', 'Statistic / samples', 'Recorded validation (unverified here)', 'Raw eligibility', 'Evidence and original values'], filtered.slice(0, limit).map(record => {
      const context = record.context;
      return [record.suite_or_lane,
        previewOperation(context),
        previewScale(context), record.timing_scope || 'Unspecified scope (not fit)', previewDuration(record),
        `${record.statistic ?? 'Raw field only'} · samples: ${record.sample_count ?? 'not recorded'}`,
        `${record.validation_status} · recorded correctness: ${record.correctness_status}`,
        `${previewValue(record.raw_performance_eligible)}; effective: false`, detail(`${record.timing_key} · ${record.raw_pointer}`, record)];
    })));
    if (filtered.length > limit) {
      const more = h('button', { type: 'button' }, `Show all ${filtered.length} timing fields`);
      more.addEventListener('click', () => { limit = Number.MAX_SAFE_INTEGER; renderRecords(); });
      records.appendChild(more);
    }
  };
  select.addEventListener('change', () => { limit = 200; renderRecords(); });
  search.addEventListener('input', () => { limit = 200; renderRecords(); });
  renderRecords();
  root.appendChild(records);
  return root;
}
export function renderDiagnosticPreview(): HTMLElement {
  const section = h('details', { id: 'diagnostic-preview', class: 'diagnostic-preview' });
  section.appendChild(h('summary', {}, 'Local benchmark update preview'));
  section.appendChild(h('p', {}, 'Open an exported preview_data.json to inspect registered, expanded, and independent timings without changing the canonical dashboard. The file stays in this browser and is not uploaded.'));
  section.appendChild(h('label', { for: 'preview-file' }, 'Choose diagnostic preview JSON:'));
  const input = h('input', { id: 'preview-file', type: 'file', accept: '.json,application/json' }) as HTMLInputElement;
  section.appendChild(input);
  const reset = h('button', { type: 'button' }, 'Clear local preview');
  section.appendChild(reset);
  const status = h('p', { role: 'status', 'aria-live': 'polite' });
  const output = h('div', { id: 'preview-output' });
  section.appendChild(status);
  section.appendChild(output);
  let epoch = 0;
  reset.addEventListener('click', () => { epoch += 1; input.value = ''; clear(output); status.textContent = 'Local preview cleared. Canonical data is unchanged.'; });
  input.addEventListener('change', async () => {
    const file = input.files?.[0];
    if (!file) return;
    const current = ++epoch;
    clear(output);
    status.textContent = 'Reading local preview…';
    try {
      if (file.size > 64 * 1024 * 1024) throw new Error('Choose a preview smaller than 64 MiB.');
      const parsed = parseDiagnosticPreview(await file.text());
      if (current !== epoch) return;
      output.appendChild(renderLoaded(parsed, file.name));
      status.textContent = `Loaded ${parsed.records.length} timing fields. Canonical data is unchanged.`;
    } catch (error) {
      if (current !== epoch) return;
      status.textContent = `Preview not loaded: ${error instanceof Error ? error.message : String(error)}`;
    }
  });
  return section;
}
