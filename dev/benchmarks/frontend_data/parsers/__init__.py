from __future__ import annotations
"""Benchmark data parsers."""

from .penalized_glm import parse_penalized_glm_bench_perf
from .solver import parse_glm_solver_benchmark
from .elasticnet import parse_elasticnet_benchmark_full
from .coxph import parse_coxph_efron_bench
from .validation import parse_comprehensive_validation, parse_coxph_package_comparison
from .cv_models import parse_lassocv_combined
from .cv_package import parse_cv_benchmark
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
from .panel_stage_b import parse_panel_stage_b_physical_validation
from .panel_stage_c import (
    parse_panel_stage_c_physical_validation,
    parse_panel_stage_c_performance,
)
from .panel_stage_c_rank_policy import (
    parse_panel_stage_c_rank_policy_physical_validation,
    parse_panel_stage_c_rank_policy_performance,
)
from .panel_stage_c_rank_df import (
    parse_panel_stage_c_rank_df_physical_validation,
    parse_panel_stage_c_rank_df_performance,
)
from .panel_stage_c_identifiability import (
    parse_panel_stage_c_identifiability_physical_validation,
    parse_panel_stage_c_identifiability_performance,
)

__all__ = [
    "parse_penalized_glm_bench_perf",
    "parse_glm_solver_benchmark",
    "parse_elasticnet_benchmark_full",
    "parse_coxph_efron_bench",
    "parse_comprehensive_validation",
    "parse_coxph_package_comparison",
    "parse_lassocv_combined",
    "parse_cv_benchmark",
    "parse_knockoff_benchmark",
    "parse_loss_functions_benchmark",
    "parse_ordered_inference_benchmark",
    "parse_pr74_inference_benchmark",
    "parse_unsupervised_benchmark",
    "parse_new_modules_benchmark",
    "parse_new_modules_with_anova_benchmark",
    "parse_p2_benchmark",
    "parse_panel_stage_b_physical_validation",
    "parse_panel_stage_c_physical_validation",
    "parse_panel_stage_c_performance",
    "parse_panel_stage_c_rank_policy_physical_validation",
    "parse_panel_stage_c_rank_policy_performance",
    "parse_panel_stage_c_rank_df_physical_validation",
    "parse_panel_stage_c_rank_df_performance",
    "parse_panel_stage_c_identifiability_physical_validation",
    "parse_panel_stage_c_identifiability_performance",
]
