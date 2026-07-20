from __future__ import annotations
"""Benchmark data parsers."""

from .penalized_glm import parse_penalized_glm_bench_perf
from .solver import parse_glm_solver_benchmark
from .elasticnet import parse_elasticnet_benchmark_full
from .coxph import parse_coxph_efron_bench
from .validation import parse_comprehensive_validation, parse_coxph_package_comparison
from .cv_models import parse_lassocv_combined
from .knockoff import parse_knockoff_benchmark
from .loss_functions import parse_loss_functions_benchmark
from .domains import (
    parse_ordered_inference_benchmark,
    parse_new_modules_benchmark,
    parse_p2_benchmark,
)
from .unsupervised import parse_unsupervised_benchmark
from .new_modules_complete import parse_new_modules_with_anova_benchmark
from .pr74_complete import parse_pr74_inference_benchmark

__all__ = [
    "parse_penalized_glm_bench_perf",
    "parse_glm_solver_benchmark",
    "parse_elasticnet_benchmark_full",
    "parse_coxph_efron_bench",
    "parse_comprehensive_validation",
    "parse_coxph_package_comparison",
    "parse_lassocv_combined",
    "parse_knockoff_benchmark",
    "parse_loss_functions_benchmark",
    "parse_ordered_inference_benchmark",
    "parse_pr74_inference_benchmark",
    "parse_unsupervised_benchmark",
    "parse_new_modules_benchmark",
    "parse_new_modules_with_anova_benchmark",
    "parse_p2_benchmark",
]
