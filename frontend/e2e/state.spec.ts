import { expect, test } from '@playwright/test';
import type { Environment, Run } from '../src/schema';
import { createDefaultState } from '../src/state';

const environments: Environment[] = [
  { env_id: 'remote-p100', label: 'P100', gpu: 'P100', cpu: 'Xeon' },
  { env_id: 'cpu-only', label: 'CPU', gpu: 'none', cpu: 'EPYC' },
];

function makeRun(envId: string, categoryIds: string[]): Run {
  return {
    run_id: `run-${envId}`,
    env_id: envId,
    category_ids: categoryIds,
    model_id: 'ExampleModel',
    comparison_id: 'example-comparison',
    case_id: 'default',
    method_config_id: 'default',
    framework: 'statgpu',
    backend: 'numpy',
    scale: {
      scale_key: 'n10_p2',
      n_samples: 10,
      n_features: 2,
      label: '10×2',
    },
    source: {
      source_id: 'example-20260101-000000000000',
      file: 'example.json',
      date: '2026-01-01',
      parser: 'test',
      parser_version: '1.0',
    },
    metrics: {},
  };
}

test('default state skips preferred environments with no runs', () => {
  const state = createDefaultState(environments, [makeRun('cpu-only', ['survival'])]);

  expect(state.selectedEnvId).toBe('cpu-only');
  expect([...state.selectedCategoryIds]).toEqual(['survival']);
  expect(state.chartViewMode).toBe('focused');
});

test('default state prefers penalized GLM on the preferred populated environment', () => {
  const runs = [
    makeRun('cpu-only', ['survival']),
    makeRun('remote-p100', ['glm', 'penalized_glm']),
  ];
  const state = createDefaultState(environments, runs);

  expect(state.selectedEnvId).toBe('remote-p100');
  expect([...state.selectedCategoryIds]).toEqual(['penalized_glm']);
  expect(state.chartViewMode).toBe('focused');
  expect(state.speedupChartLimit).toBe(100);
});
