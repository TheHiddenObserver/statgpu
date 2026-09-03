import type {
  BenchmarkData,
  ParseReport,
  SourceInventory,
} from '../schema';

export const BENCHMARK_SCHEMA_VERSION = '1.1.0';
export const PARSE_REPORT_VERSION = '2.0';
export const SOURCE_INVENTORY_VERSION = '2.0';

export interface BenchmarkBundle {
  data: BenchmarkData;
  parseReport: ParseReport | null;
  sourceInventory: SourceInventory | null;
}

export interface BenchmarkLoadOptions {
  signal?: AbortSignal;
}

/**
 * Stable dashboard boundary. A future API-backed provider must return the same
 * normalized bundle and preserve the schema/version and generation-id checks.
 */
export interface BenchmarkDataProvider {
  loadBundle(options?: BenchmarkLoadOptions): Promise<BenchmarkBundle>;
  clearCache(): void;
}

export interface StaticJsonBenchmarkProviderOptions {
  baseUrl: string;
  fetcher?: typeof fetch;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : null;
}

function validateBenchmarkData(value: unknown): BenchmarkData {
  const record = asRecord(value);
  if (!record) throw new Error('Benchmark data is not a JSON object');
  if (record.schema_version !== BENCHMARK_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported schema ${String(record.schema_version)}; expected ${BENCHMARK_SCHEMA_VERSION}`,
    );
  }
  const meta = asRecord(record.meta);
  if (!meta || typeof meta.generation_id !== 'string') {
    throw new Error('Benchmark data is missing meta.generation_id');
  }
  if (!Array.isArray(record.runs)) {
    throw new Error('Benchmark data is missing the runs array');
  }
  return value as BenchmarkData;
}

function validateParseReport(value: unknown): ParseReport | null {
  const record = asRecord(value);
  if (
    !record ||
    record.report_version !== PARSE_REPORT_VERSION ||
    typeof record.generation_id !== 'string'
  ) {
    return null;
  }
  return value as ParseReport;
}

function validateSourceInventory(value: unknown): SourceInventory | null {
  const record = asRecord(value);
  if (
    !record ||
    record.inventory_version !== SOURCE_INVENTORY_VERSION ||
    typeof record.generation_id !== 'string'
  ) {
    return null;
  }
  const requiredCounts = [
    'discovered_json_artifacts',
    'classified_candidate_sources',
    'eligible_sources',
    'registered_sources',
    'available_registered_sources',
    'parsed_registered_sources',
    'eligible_unregistered_sources',
    'not_canonical_ready_sources',
    'historical_or_excluded_sources',
    'unclassified_artifacts',
  ];
  if (
    requiredCounts.some(
      key => !Number.isInteger(record[key]) || Number(record[key]) < 0,
    )
  ) {
    return null;
  }
  return value as SourceInventory;
}

export class StaticJsonBenchmarkProvider implements BenchmarkDataProvider {
  private readonly baseUrl: string;
  private readonly fetcher: typeof fetch;
  private cachedBundle: Promise<BenchmarkBundle> | null = null;

  constructor(options: StaticJsonBenchmarkProviderOptions) {
    this.baseUrl = options.baseUrl.endsWith('/')
      ? options.baseUrl
      : `${options.baseUrl}/`;
    this.fetcher = options.fetcher ?? globalThis.fetch.bind(globalThis);
  }

  loadBundle(options: BenchmarkLoadOptions = {}): Promise<BenchmarkBundle> {
    // An AbortSignal is request-scoped and must not poison the shared cache.
    if (options.signal) return this.loadUncached(options.signal);
    if (!this.cachedBundle) this.cachedBundle = this.loadUncached();
    return this.cachedBundle;
  }

  clearCache(): void {
    this.cachedBundle = null;
  }

  private async requiredJson(path: string, signal?: AbortSignal): Promise<unknown> {
    const response = await this.fetcher(`${this.baseUrl}${path}`, { signal });
    if (!response.ok) {
      throw new Error(`Failed to load ${path}: ${response.status}`);
    }
    return response.json();
  }

  private async optionalJson(path: string, signal?: AbortSignal): Promise<unknown | null> {
    try {
      const response = await this.fetcher(`${this.baseUrl}${path}`, { signal });
      if (!response.ok) return null;
      return response.json();
    } catch (error) {
      if (signal?.aborted) throw error;
      return null;
    }
  }

  private async loadUncached(signal?: AbortSignal): Promise<BenchmarkBundle> {
    const [dataValue, reportValue, inventoryValue] = await Promise.all([
      this.requiredJson('data/benchmark_data.json', signal),
      this.optionalJson('data/parse_report.json', signal),
      this.optionalJson('data/source_inventory.json', signal),
    ]);
    const data = validateBenchmarkData(dataValue);
    let parseReport = validateParseReport(reportValue);
    let sourceInventory = validateSourceInventory(inventoryValue);
    if (parseReport?.generation_id !== data.meta.generation_id) parseReport = null;
    if (sourceInventory?.generation_id !== data.meta.generation_id) {
      sourceInventory = null;
    }
    return { data, parseReport, sourceInventory };
  }
}
