import { expect, test } from '@playwright/test';
import type { Environment, Run } from '../src/schema';
import { getUniqueScaleKeys } from '../src/scales';
import {
  createDefaultState,
  setSelectedModel,
  setSelectedPenalty,
  setSelectedSolver,
  setSelectedVariant,
  toggleScaleKey,
} from '../src/state';

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
  expect(state.selectedMetricScope).toBe('all');
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
  expect(state.selectedMetricScope).toBe('all');
  expect(state.chartViewMode).toBe('focused');
  expect(state.timingChartGroupLimit).toBe(Number.MAX_SAFE_INTEGER);
  expect(state.speedupChartLimit).toBe(Number.MAX_SAFE_INTEGER);
});

test('scale keys are ordered by numeric workload dimensions, not lexicographically', () => {
  const base = makeRun('remote-p100', ['linear_models']);
  const runs: Run[] = [
    {
      ...base,
      run_id: 'run-20k',
      scale: {
        scale_key: 'n20000_p5',
        n_samples: 20000,
        n_features: 5,
        label: '20K×5',
      },
    },
    {
      ...base,
      run_id: 'run-1k',
      scale: {
        scale_key: 'n1000_p50',
        n_samples: 1000,
        n_features: 50,
        label: '1K×50',
      },
    },
    {
      ...base,
      run_id: 'run-5k',
      scale: {
        scale_key: 'n5000_p10',
        n_samples: 5000,
        n_features: 10,
        label: '5K×10',
      },
    },
    {
      ...base,
      run_id: 'run-1k-duplicate',
      scale: {
        scale_key: 'n1000_p50',
        n_samples: 1000,
        n_features: 50,
        label: '1K×50',
      },
    },
  ];

  expect(getUniqueScaleKeys(runs)).toEqual([
    'n1000_p50',
    'n5000_p10',
    'n20000_p5',
  ]);
});

test('upstream filter changes clear backend and external selections', () => {
  const state = createDefaultState(environments, [
    makeRun('remote-p100', ['penalized_glm']),
  ]);

  const seedDownstream = () => {
    state.selectedScaleKeys.add('n10_p2');
    state.selectedBackends.add('cupy');
    state.showExternal.add('sklearn');
  };
  const expectDownstreamCleared = () => {
    expect(state.selectedScaleKeys.size).toBe(0);
    expect(state.selectedBackends.size).toBe(0);
    expect(state.showExternal.size).toBe(0);
  };

  seedDownstream();
  setSelectedModel(state, 'ModelA');
  expectDownstreamCleared();

  seedDownstream();
  setSelectedVariant(state, 'variant-a');
  expectDownstreamCleared();

  seedDownstream();
  setSelectedPenalty(state, 'l1');
  expectDownstreamCleared();

  seedDownstream();
  setSelectedSolver(state, 'fista');
  expectDownstreamCleared();

  state.selectedBackends.add('torch');
  state.showExternal.add('statsmodels');
  toggleScaleKey(state, 'n20_p2');
  expect(state.selectedScaleKeys).toEqual(new Set(['n20_p2']));
  expect(state.selectedBackends.size).toBe(0);
  expect(state.showExternal.size).toBe(0);
});
