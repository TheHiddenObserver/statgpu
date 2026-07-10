"""Parser registry — hardcoded in A1, manifest-driven in A2."""

from .parsers import (
    parse_penalized_glm_bench_perf,
    parse_glm_solver_benchmark,
    parse_elasticnet_benchmark_full,
)

PARSER_REGISTRY: dict[str, dict] = {
    "penalized_glm_bench_perf_2026-06-22.json": {
        "parser": parse_penalized_glm_bench_perf,
        "env_id": "remote-p100",
    },
    "glm_solver_benchmark_2026-06-23.json": {
        "parser": parse_glm_solver_benchmark,
        "env_id": "remote-p100",
    },
    "benchmark_full/benchmark_statgpu_all.json": {
        "parser": parse_elasticnet_benchmark_full,
        "env_id": "remote-p100",
    },
    "benchmark_full/benchmark_glmnet_all.json": {
        "parser": parse_elasticnet_benchmark_full,
        "env_id": "remote-p100",
    },
}
