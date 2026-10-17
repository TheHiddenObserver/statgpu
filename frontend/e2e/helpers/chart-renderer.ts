/// <reference types="vite/client" />
/** Exercise application renderers in Node; only ECharts' rendering sink is replaced. */
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import type * as Timing from '../../src/charts/TimingChart';
import type * as Fallback from '../../src/components/ChartDataFallback';
import type * as Data from '../../src/data';
import type * as State from '../../src/state';
import type * as Provider from '../../src/providers/benchmark';
import type * as Convergence from '../../src/components/panels/ConvergencePanel';

export interface TooltipParameter {
  seriesName: string;
  value: number | null;
  color: string;
  axisValueLabel?: string;
  dataIndex?: number;
}

export interface CapturedTimingOption {
  xAxis: { data: string[] };
  series: { name: string; data: (number | null)[] }[];
  tooltip: { formatter: (params: TooltipParameter[]) => string };
}

type Renderer = typeof Timing & typeof Fallback & typeof Data & typeof State & typeof Provider & typeof Convergence & {
  capturedOptions: CapturedTimingOption[];
};

export async function loadChartRenderer(): Promise<Renderer> {
  const bundle = await build({
    stdin: {
      contents: `
        export * from './src/charts/TimingChart';
        export * from './src/components/ChartDataFallback';
        export * from './src/components/panels/ConvergencePanel';
        export * from './src/data';
        export * from './src/state';
        export * from './src/providers/benchmark';
        export { capturedOptions } from './src/echarts';
      `,
      resolveDir: fileURLToPath(new URL('../../', import.meta.url)),
      loader: 'ts',
    },
    bundle: true,
    platform: 'node',
    format: 'esm',
    write: false,
    define: { 'import.meta.env.BASE_URL': JSON.stringify('/') },
    plugins: [{
      name: 'capture-echarts-options',
      setup(builder) {
        builder.onResolve({ filter: /(?:^|\/)echarts$/ }, () => ({
          path: 'capture-echarts', namespace: 'test',
        }));
        builder.onLoad({ filter: /.*/, namespace: 'test' }, () => ({
          contents: `
            export const capturedOptions = [];
            export const echarts = {
              getInstanceByDom: () => undefined,
              init: () => ({ clear() {}, setOption(option) { capturedOptions.push(option); } }),
            };
          `,
          loader: 'js',
        }));
      },
    }],
  });
  return import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
}

/** Minimal text-node DOM for the actual lazy fallback renderer, not a table reimplementation. */
export class TestElement {
  readonly attributes: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  readonly children: TestElement[] = [];
  readonly events = new Map<string, () => void>();
  open = false;

  constructor(readonly tag: string, readonly text = '') {}
  setAttribute(name: string, value: string): void { this.attributes[name] = value; }
  append(...children: TestElement[]): void { this.children.push(...children); }
  appendChild(child: TestElement): TestElement { this.append(child); return child; }
  addEventListener(name: string, callback: () => void): void { this.events.set(name, callback); }
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join(''); }
  findAll(predicate: (element: TestElement) => boolean): TestElement[] {
    return [this, ...this.children.flatMap(child => child.findAll(() => true))].filter(predicate);
  }
}

export function withTestDocument<T>(callback: () => T): T {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {
      createElement: (tag: string) => new TestElement(tag),
      createTextNode: (text: string) => new TestElement('#text', text),
    },
  });
  try { return callback(); }
  finally {
    if (previous) Object.defineProperty(globalThis, 'document', previous);
    else Reflect.deleteProperty(globalThis, 'document');
  }
}
