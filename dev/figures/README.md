# Clustering teaching figures

These are scientific illustrations generated from numeric data, not decorative
images or stored estimator training histories. Runtime modules and existing tests
are unchanged. Both language variants use the same data, fitted parameters,
geometry and scales; only labels differ. SVG fonts are embedded as paths.

## Regenerate

From the repository root:

```sh
python dev/figures/generate_clustering.py
python -m pytest -q dev/tests/test_clustering_teaching_figures.py
```

Requirements: the local `statgpu` package and its NumPy/SciPy dependencies,
Matplotlib, and Noto Sans CJK Regular at
`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` for Chinese labels.
Use an installed official font package or adjust `clustering_common.FONT` to an
equivalent installed Noto CJK font. Rendering tests skip if optional Matplotlib or
Chinese fonts are absent; asset/placement tests still run. Regeneration was
validated with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8.
Byte determinism is checked within the same environment; different renderer/font
versions can change SVG bytes. SVG IDs have a fixed salt and metadata has no date.
`--output-dir PATH` writes a separate candidate set for comparison.

## Data and meaning

- **KMeans:** Eight explicit points and chosen initial centers. A single NumPy
  nearest-center assignment and mean update gives a fixed-assignment squared
  error of 11 then 4.75. This is an analytical Lloyd update illustration, not a
  trace from `KMeans.fit` (which exposes no public per-iteration center history).
- **GaussianMixture:** 220 overlapping observations, RNG seed 20261014, an actual
  CPU full-covariance two-component fit with seed 17, five starts, 500 iterations,
  tolerance 1e-7 and covariance regularization 1e-6. Display components are sorted
  by their first mean coordinate. `predict_proba` is checked against independently
  calculated weighted Gaussian densities. Ellipses have Mahalanobis radii 1 and
  2 in the fitted component distribution; they are not confidence intervals.
- **DBSCAN:** Fifteen explicit points, `eps=0.52`, `min_samples=5` including self.
  Independent closed-neighborhood counts and breadth-first core connectivity
  establish 10 core, 2 unambiguous border and 3 noise points; partition and core
  indices are checked against the actual CPU fit. This fixture contains connected
  cores and does not make a claim about every possible configuration.
- **AgglomerativeClustering:** Six explicit numbered points; actual CPU Ward
  `children_` and `distances_`. Every merge minimizes independently computed
  within-cluster squared-error increase, and its height equals sqrt(2 ΔSSE).
  Recursive leaf counts form the plotted tree. Actual two- and three-cluster fits
  agree with cuts between adjacent heights; no tied heights occur.
- **MiniBatchKMeans:** Four explicit unweighted batches, fixed initial centers,
  seed 17 and four real CPU `partial_fit` calls. Public centers/counts are checked
  against independently accumulated assignments, sums and counts. Counts progress
  from [0,0] through [3,2], [5,6], [9,8] to [12,12]. Gray observations show all
  batches, without claiming they are latest-batch `labels_` or full-data predictions.
  There is no automatic forgetting. This is not the behavior of batch `fit`.

Every 2D distance panel uses equal axis scaling. Discrete membership uses color
and shape together; core/border/noise use filled/hollow/cross markers. GMM
probabilities use a continuous sequential 0–1 scale. Label IDs carry no ordering
or population meaning. The figures show CPU teaching examples, not GPU execution,
performance, inference coverage or cross-backend bitwise agreement.

## Validation and visual review

The new tests check source data, independent scientific identities, rejection of
corrupted outputs, bilingual references/meaningful alt text, position after
mathematics and before learner examples, safe SVG structure and repeatable bytes.
Always also inspect every language SVG after rasterization at approximately
780px content width, including text, legends, colorbars and clipping. Review
surrounding prose at the same width. Browser screenshots, when available, check
the actual embedding; a composed same-width preview is only a layout aid and
must not be represented as a browser screenshot.
