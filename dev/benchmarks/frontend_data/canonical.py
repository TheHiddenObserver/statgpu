"""Canonicalization helpers, constants, and maps for benchmark data generation."""

import json
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Known types
# ---------------------------------------------------------------------------

KNOWN_FAMILIES = [
    "squared_error", "logistic", "poisson", "gamma",
    "inverse_gaussian", "negative_binomial", "tweedie",
]

KNOWN_PENALTIES = [
    "none", "l1", "l2", "elasticnet", "scad", "mcp", "adaptive_l1",
    "group_lasso", "group_mcp", "group_scad",
]

# ---------------------------------------------------------------------------
# Name maps
# ---------------------------------------------------------------------------

BACKEND_MAP = {
    "numpy": "numpy", "cpu": "numpy", "statgpu_cpu": "numpy",
    "numPy": "numpy", "NumPy": "numpy", "CPU/NumPy": "numpy",
    "cupy": "cupy", "cuda": "cupy", "statgpu_gpu_cupy": "cupy",
    "CuPy": "cupy", "cupy_gpu": "cupy",
    "torch": "torch", "statgpu_gpu_torch": "torch",
    "Torch": "torch", "torch_cuda": "torch",
}

FRAMEWORK_MAP = {
    "statgpu": "statgpu", "statgpu_cpu": "statgpu",
    "statgpu_gpu_cupy": "statgpu", "statgpu_gpu_torch": "statgpu",
    "sklearn": "sklearn", "scikit-learn": "sklearn",
    "sklearn.linear_model": "sklearn",
    "statsmodels": "statsmodels", "sm": "statsmodels",
    "glmnet": "glmnet", "R": "r", "r": "r",
}

SCALE_CONFIG = {
    "small_5k": {"n_samples": 5000, "n_features": 500, "label": "5K×500"},
    "medium_100k": {"n_samples": 100000, "n_features": 50, "label": "100K×50"},
}

SOLVER_KIND_MAP = {
    "auto": "dispatch",
    "exact": "manual", "newton": "manual", "irls": "manual",
    "lbfgs": "manual", "fista": "manual", "fista_bb": "manual",
    "admm": "manual", "irls_cd": "manual",
    "proximal_irls_cd": "manual", "proximal_newton": "manual",
}

SOLVER_DISPLAY_MAP = {
    "auto": "Auto (best)", "exact": "Exact", "newton": "Newton",
    "irls": "IRLS", "lbfgs": "L-BFGS", "fista": "FISTA",
    "fista_bb": "FISTA-BB", "admm": "ADMM",
}

FAMILY_MODEL_MAP = {
    "squared_error": "PenalizedLinearRegression",
    "logistic": "PenalizedLogisticRegression",
    "poisson": "PenalizedPoissonRegression",
    "gamma": "PenalizedGammaRegression",
    "inverse_gaussian": "PenalizedInverseGaussianRegression",
    "negative_binomial": "PenalizedNegativeBinomialRegression",
    "tweedie": "PenalizedTweedieRegression",
}

SPEEDUP_REFERENCE_BY_SOURCE = {
    "glm_solver_benchmark_2026-06-23.json": {
        "reference_backend": "numpy",
        "reference_framework": "statgpu",
        "reported_semantics": "reported_by_runner",
    }
}

CATEGORIES = [
    {"category_id": "penalized_glm", "name_zh": "惩罚GLM", "name_en": "Penalized GLM"},
    {"category_id": "linear_models", "name_zh": "线性模型", "name_en": "Linear Models"},
    {"category_id": "glm", "name_zh": "GLM", "name_en": "GLM"},
    {"category_id": "survival", "name_zh": "生存分析", "name_en": "Survival Analysis"},
    {"category_id": "robust_quantile", "name_zh": "稳健/分位数", "name_en": "Robust/Quantile"},
    {"category_id": "unsupervised", "name_zh": "无监督学习", "name_en": "Unsupervised"},
    {"category_id": "ordered", "name_zh": "有序模型", "name_en": "Ordered Models"},
    {"category_id": "nonparametric", "name_zh": "非参数", "name_en": "Nonparametric"},
    {"category_id": "panel", "name_zh": "面板数据", "name_en": "Panel Data"},
    {"category_id": "covariance", "name_zh": "协方差估计", "name_en": "Covariance"},
    {"category_id": "feature_selection", "name_zh": "特征选择", "name_en": "Feature Selection"},
    {"category_id": "anova", "name_zh": "ANOVA", "name_en": "ANOVA"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_hash(s: str, length: int = 6) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:length]


def make_run_id(
    model_id: str, loss: str, penalty: str, solver: str,
    backend: str, framework: str, scale_key: str,
    env_id: str, session_id: str, source_hash: str,
) -> str:
    parts = [model_id, loss, penalty or "none", solver, backend or "ext",
             framework, scale_key, env_id, session_id[:8] if session_id else "", source_hash]
    return "-".join(str(p) for p in parts).lower().replace(" ", "")


def make_scale_key(n_samples: int, n_features: int) -> str:
    return f"n{n_samples}_p{n_features}"


def make_scale_label(n_samples: int, n_features: int) -> str:
    if n_samples >= 1000:
        ns = f"{n_samples//1000}K" if n_samples % 1000 == 0 else f"{n_samples/1000:.0f}K"
    else:
        ns = str(n_samples)
    return f"{ns}×{n_features}"


def parse_family_penalty_solver(key: str):
    """Parse a key like 'squared_error_l1_auto' into (family, penalty, solver)."""
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in SOLVER_KIND_MAP:
        solver = parts[1]
        prefix = parts[0]
    else:
        solver = "auto"
        prefix = key

    for fam in sorted(KNOWN_FAMILIES, key=len, reverse=True):
        if prefix.startswith(fam):
            family = fam
            penalty = prefix[len(fam) + 1:]
            if penalty not in KNOWN_PENALTIES:
                penalty = "none"
            return family, penalty, solver

    for pen in sorted(KNOWN_PENALTIES, key=len, reverse=True):
        if prefix.endswith("_" + pen):
            penalty = pen
            family = prefix[:-(len(pen) + 1)]
            return family, penalty, solver

    return prefix, "none", solver


def normalize_utf8_bytes(raw: bytes) -> bytes:
    """Normalize UTF-8 bytes for consistent SHA256."""
    if raw.startswith(b'\xef\xbb\xbf'):
        raise ValueError("BOM not allowed in source file")
    text = raw.decode("utf-8")
    if '\r' in text.replace('\r\n', ''):
        raise ValueError("Bare CR not allowed in source file")
    return text.replace('\r\n', '\n').encode("utf-8")


def source_sha256(path: Path) -> str:
    """Compute SHA256 of a file after UTF-8/line-ending normalization."""
    return hashlib.sha256(normalize_utf8_bytes(path.read_bytes())).hexdigest()
