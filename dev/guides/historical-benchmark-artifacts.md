# Historical benchmark artifact index

This is a developer provenance index of April–May 2026 experiments previously listed in the user benchmark pages. It is not a current capability or backend recommendation. Artifact names alone do not establish the executed backend, synchronization, precision, hardware, or source revision. Do not pool these runs with later dashboard sources.

Some paths below refer to old remote outputs that are not present in this checkout. They are retained as historical identifiers, not working download links or verified measurements. Present artifacts are linked.

## Remote Phase 2 artifacts

- `results/unsupervised_phase2_remote_20260502_142727.json` (not present in the reviewed checkout)
- `results/unsupervised_phase2_full_comparison_20260502_142727.md` (not present in the reviewed checkout)
- `results/unsupervised_phase2_dbscan_cython_benchmark_20260502_153927.json` (not present in the reviewed checkout)
- `results/unsupervised_phase2_dbscan_cython_summary_20260502_153927.md` (not present in the reviewed checkout)
- `results/unsupervised_phase2_dbscan_cython_final_20260502_160719.json` (not present in the reviewed checkout)
- `results/unsupervised_phase2_final_20260502_160719.json` (not present in the reviewed checkout)
- `results/unsupervised_phase2_final_summary_20260502_160719.md` (not present in the reviewed checkout)
- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json` (not present in the reviewed checkout)
- `results/unsupervised_phase2_verify_20260502_210000.json` (not present in the reviewed checkout)
- `results/unsupervised_phase2_verify_summary_20260502_210000.md` (not present in the reviewed checkout)

## Remote Phase 3 artifacts

- [results/unsupervised_phase3_remote_finalopt_20260505_084444.json](../../results/unsupervised_phase3_remote_finalopt_20260505_084444.json)
- [results/unsupervised_phase3_remote_finalopt_20260505_084444.md](../../results/unsupervised_phase3_remote_finalopt_20260505_084444.md)
- [results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.json](../../results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.json)
- [results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.md](../../results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.md)
- [results/unsupervised_phase3_remote_perfopt2_large_tabular_20260505_132223.json](../../results/unsupervised_phase3_remote_perfopt2_large_tabular_20260505_132223.json)
- [results/unsupervised_phase3_remote_perfopt2_large_tabular_bs4096_20260505_132359.json](../../results/unsupervised_phase3_remote_perfopt2_large_tabular_bs4096_20260505_132359.json)

## Remote Phase 3B artifacts

- [results/unsupervised_phase3b_verify_20260507_003957.json](../../results/unsupervised_phase3b_verify_20260507_003957.json)
- [results/unsupervised_phase3b_verify_summary_20260507_003957.md](../../results/unsupervised_phase3b_verify_summary_20260507_003957.md)

## Remote Phase 3C artifacts

- [results/unsupervised_phase3c_opt7_20260507_185500.json](../../results/unsupervised_phase3c_opt7_20260507_185500.json)
- [results/unsupervised_phase3c_opt7_summary_20260507_185500.md](../../results/unsupervised_phase3c_opt7_summary_20260507_185500.md)
- [results/unsupervised_phase3c_opt7_large_bs4096_20260507_185500.json](../../results/unsupervised_phase3c_opt7_large_bs4096_20260507_185500.json)
- [results/unsupervised_phase3c_opt7_large_bs4096_summary_20260507_185500.md](../../results/unsupervised_phase3c_opt7_large_bs4096_summary_20260507_185500.md)
- [results/unsupervised_phase3c_opt7_xlarge_20260507_185500.json](../../results/unsupervised_phase3c_opt7_xlarge_20260507_185500.json)
- [results/unsupervised_phase3c_opt7_xlarge_summary_20260507_185500.md](../../results/unsupervised_phase3c_opt7_xlarge_summary_20260507_185500.md)

## Remote supplement artifacts

- `results/remote_fisher_cauchy_benchmark_2026-04-05.json` (not present in the reviewed checkout)
- `results/remote_fisher_cauchy_benchmark_2026-04-05.md` (not present in the reviewed checkout)

## Covariance comparison reported on 2026-04-10

The old user index named `tmp_remote_covariance_full_compare.py` and `results/remote_covariance_full_compare_2026-04-10.json`, with HC2/HC3/HAC, linear `n=8000, p=24`, logistic `n=12000, p=16`, and two timed repeats after warmup. The named result artifact is not present in the reviewed checkout, so its timings and precision summary are not retained as current user-facing evidence. Recover the original source, environment, and validator metadata before reusing those claims.

## ElasticNet experiments reported on 2026-04-18

- [Small-workload JSON](../../results/benchmark_elasticnet_sklearn_2026-04-18.json)
- [Large-workload JSON](../../results/large_scale/benchmark_elasticnet_large_scale_2026-04-18.json)
- [R glmnet JSON](../../results/benchmark_full/benchmark_glmnet_all.json)
- [Paired statgpu JSON](../../results/benchmark_full/benchmark_statgpu_all.json)
- `results/benchmark_full/benchmark_complete_report.md` (not present in the reviewed checkout)
- [Historical combined summary](../../results/benchmark_complete_summary.md)
- [Historical small-workload summary](../../results/benchmark_elasticnet_sklearn_2026-04-18.md)
- [Historical large-workload summary](../../results/large_scale/benchmark_elasticnet_large_scale_2026-04-18.md)

Associated historical reproduction scripts:

- [R glmnet runner](../benchmarks/benchmark_glmnet_full.R)
- [Paired statgpu runner](../benchmarks/benchmark_statgpu_full.py)
- [Combined remote runner](../benchmarks/run_full_benchmark.py)
- [Large-scale remote runner](../benchmarks/run_large_scale.py)

The old summaries contain backend recommendations and precision claims that go beyond the recorded evidence. The JSON does not identify the executed hardware, software versions, source commit, or repeat distribution. In the associated large-scale runner, the row labelled `statgpu_gpu_torch` constructs `device='cuda'`; its label is therefore not proof of Torch execution. The large-scale JSON stores coefficient norms rather than full coefficient vectors, so agreement of those norms is not a componentwise coefficient check. For reader-facing interpretation, see the [historical ElasticNet appendix](../../docs/en/benchmarks.md#elastic-net-benchmarks).
