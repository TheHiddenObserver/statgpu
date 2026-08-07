/** Muted, color-blind-conscious palette for backends and reference frameworks. */
export const COLORS: Record<string, string> = {
  numpy: '#5f72c4',
  cupy: '#86b86f',
  torch: '#d8ad58',
  sklearn: '#ca7278',
  statsmodels: '#67a8be',
  scipy: '#9b78bd',
  linearmodels: '#5f9f8c',
  pygam: '#b07b9f',
  glmnet: '#4f9d76',
  r: '#4f9d76',
};

export const CHART_STYLE = {
  text: '#4c5567',
  muted: '#7b8494',
  axis: '#c8cfdb',
  grid: '#e8ecf3',
  parity: '#697386',
  tooltipBackground: 'rgba(31, 36, 58, 0.94)',
  speedupComputed: '#579b70',
  speedupReported: '#7eb48e',
  slowdownComputed: '#c96f73',
  slowdownReported: '#da9295',
} as const;

/** Quality badge colors */
export const QUALITY_COLORS: Record<string, string> = {
  measured: '#579b70',
  reported: '#c79136',
  computed: '#5876c5',
  partial: '#c96f73',
};
