import { expect, test } from '@playwright/test';
import type { Environment, Run } from '../src/schema';
import { chartSolverFamilyIdentity } from '../src/identity';
import { createDefaultState } from '../src/state';
import {
  formatGroupLabel,
  selectTimingRuns,
} from '../src/charts/TimingChart';
import {
  formatRunLabel,
  selectSpeedupRuns,
} from '../src/charts/SpeedupChart';

const environments: Environment[] = [
  { env_id: 'remote-p100', label: 'P100', gpu: 'P100', cpu: 'Xeon' },
];

function makeRun(overrides: Partial<Run> & Pick<Run, 'run_id' | 'model_id'>): Run {
  return {
    env_id: 'remote-p100',
    category_ids: ['linear_models'],
    comparison_id: 'comparison-a',
    case_id: 'case-a',
    method_config_id: 'method-a',
    loss: 'squared_error',
    penalty: null,
    solver: 'newton',
    solver_display: 'Newton',
    solver_kind: 'manual',
    framework: 'statgpu',
    backend: 'numpy',
    scale: {
      scale_key: 'n20000_p5',
      n_samples: 20000,
      n_features: 5,
      label: '20K×5',
    },
    source: {
      source_id: 'source-a',
      file: 'source-a.json',
      date: '2026-07-20',
      parser: 'test',
      parser_version: '1.0',
    },
    metrics: {
      timing: {
        fit_time_ms: 10,
        quality: 'measured',
        source_file: 'source-a.json',
      },
      speedup: {
        value: 2,
        reference_backend: null,
        reference_framework: 'external',
        reported_semantics: 'reported_by_runner',
        quality: 'reported',
        source_file: 'source-a.json',
      },
    },
    ...overrides,
  };
}

test('focused selectors prefer Auto per solver family without dropping manual-only families', () => {
  const dispatch = makeRun({
    run_id: 'dispatch',
    model_id: 'AutoModel',
    solver: 'auto',
    solver_display: 'Auto (best)',
    solver_kind: 'dispatch',
  });
  const manualSibling = makeRun({
    run_id: 'manual-sibling',
    model_id: 'AutoModel',
    solver: 'fista',
    solver_display: 'FISTA',
    solver_kind: 'manual',
  });
  const manualOnly = makeRun({
    run_id: 'manual-only',
    model_id: 'ManualOnlyModel',
    case_id: 'case-b',
    method_config_id: 'method-b',
    solver: 'newton',
    solver_display: 'Newton',
    solver_kind: 'manual',
  });
  const runs = [dispatch, manualSibling, manualOnly];
  const state = createDefaultState(environments, runs);

  expect(chartSolverFamilyIdentity(dispatch, false)).toEqual(
    chartSolverFamilyIdentity(manualSibling, false),
  );

  const timingIds = selectTimingRuns(runs, state).runs.map((run) => run.run_id);
  expect(timingIds).toEqual(['dispatch', 'manual-only']);

  const speedupIds = selectSpeedupRuns(runs, state).runs.map((run) => run.run_id);
  expect(speedupIds).toEqual(['dispatch', 'manual-only']);
});

test('focused labels identify manual solvers and explicitly selected scales', () => {
  const run = makeRun({
    run_id: 'manual-label',
    model_id: 'ManualModel',
    solver: 'newton',
    solver_display: 'Newton',
    solver_kind: 'manual',
  });

  expect(formatGroupLabel(run, true, true)).toContain('Newton');
  expect(formatGroupLabel(run, true, true)).toContain('20K×5');
  expect(formatRunLabel(run, true, true)).toContain('Newton');
  expect(formatRunLabel(run, true, true)).toContain('20K×5');
});
