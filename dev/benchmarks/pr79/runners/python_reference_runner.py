#!/usr/bin/env python3
"""Python external-framework reference runner for PR79 benchmarks.

Runs statsmodels, scikit-learn, and linearmodels on the same data
as the statgpu runner, producing comparable raw JSON records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.runners.common import (
    make_case_id, make_method_config_id, make_raw_run,
    record_environment, safe_run,
)


def _get_git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Linear reference
# ---------------------------------------------------------------------------


def ref_linear_statsmodels(
    X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
    cov_type: str = "nonrobust",
) -> Dict[str, Any]:
    """LinearRegression via statsmodels OLS/WLS."""
    import statsmodels.api as sm

    if sample_weight is not None:
        model = sm.WLS(y, sm.add_constant(X), weights=sample_weight)
    else:
        model = sm.OLS(y, sm.add_constant(X))

    cov_map = {"nonrobust": "nonrobust", "hc0": "HC0", "hc1": "HC1"}
    sm_cov = cov_map.get(cov_type, "nonrobust")
    res = model.fit(cov_type=sm_cov)

    return {
        "coef_": res.params[1:].tolist(),
        "intercept_": float(res.params[0]),
        "_bse": res.bse[1:].tolist(),
        "rsquared": float(res.rsquared),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "fvalue": float(res.fvalue) if hasattr(res, "fvalue") else None,
    }


def ref_linear_sklearn(X: np.ndarray, y: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """LinearRegression via scikit-learn."""
    from sklearn.linear_model import LinearRegression as SkLinear
    model = SkLinear(fit_intercept=True)
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return {
        "coef_": model.coef_.tolist(),
        "intercept_": float(model.intercept_),
    }


def ref_ridge_sklearn(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Ridge via scikit-learn (with documented alpha mapping)."""
    from sklearn.linear_model import Ridge as SkRidge

    n = X.shape[0]
    sw_sum = float(np.sum(sample_weight)) if sample_weight is not None else float(n)
    # statgpu alpha -> sklearn alpha: multiply by sum of weights
    sk_alpha = alpha * sw_sum

    model = SkRidge(alpha=sk_alpha, fit_intercept=True, solver="cholesky")
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return {
        "coef_": model.coef_.tolist(),
        "intercept_": float(model.intercept_),
    }


# ---------------------------------------------------------------------------
# Panel reference
# ---------------------------------------------------------------------------


def ref_pooled_linearmodels(
    X: np.ndarray, y: np.ndarray, entity: np.ndarray, time_idx: np.ndarray,
    cov_type: str = "nonrobust",
) -> Dict[str, Any]:
    """PooledOLS via linearmodels."""
    import pandas as pd

    df = pd.DataFrame({
        "y": y, "x1": X[:, 0], "x2": X[:, 1] if X.shape[1] > 1 else X[:, 0],
        "entity": entity, "time": time_idx,
    })
    df = df.set_index(["entity", "time"])

    try:
        from linearmodels.panel import PooledOLS as LmPooledOLS
        exog_vars = [c for c in df.columns if c.startswith("x")]
        model = LmPooledOLS(df["y"], df[exog_vars])
        res = model.fit(cov_type=cov_type)
        return {
            "coef_": res.params.values.tolist(),
            "_bse": res.std_errors.values.tolist(),
            "rsquared": float(res.rsquared),
        }
    except ImportError:
        return {"error": "linearmodels not installed"}


# ---------------------------------------------------------------------------
# CoxPH reference
# ---------------------------------------------------------------------------


def ref_coxph_statsmodels(
    X: np.ndarray, time: np.ndarray, event: np.ndarray,
    ties: str = "efron", entry: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """CoxPH via statsmodels PHReg."""
    import statsmodels.api as sm

    model = sm.PHReg(time, sm.add_constant(X, has_constant="add"),
                     status=event, ties=ties, entry=entry)
    res = model.fit(disp=0)
    return {
        "coef_": res.params[:-1].tolist(),
        "_bse": res.bse[:-1].tolist(),
        "_log_likelihood": float(res.llf),
        "aic": float(res.aic),
    }


def ref_coxph_lifelines(
    X: np.ndarray, time: np.ndarray, event: np.ndarray,
) -> Dict[str, Any]:
    """CoxPH via lifelines."""
    try:
        from lifelines import CoxPHFitter
        import pandas as pd

        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["time"] = time
        df["event"] = event
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event")
        return {
            "coef_": cph.params_.values.tolist(),
            "_bse": cph.summary["se(coef)"].values.tolist(),
            "_log_likelihood": float(cph.log_likelihood_),
        }
    except ImportError:
        return {"error": "lifelines not installed"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all() -> List[Dict[str, Any]]:
    from dev.benchmarks.pr79.generators.linear import (
        generate_linear_full_rank, generate_linear_rank_deficient,
        generate_linear_weighted, case_params_linear,
        case_params_linear_rank_def, case_params_linear_weighted,
    )
    from dev.benchmarks.pr79.generators.survival import (
        generate_coxph_no_ties, generate_coxph_small_ties,
        case_params_coxph_no_ties, case_params_coxph_small_ties,
    )
    from dev.benchmarks.pr79.generators.panel import (
        generate_pooled_balanced, case_params_pooled,
    )

    env = record_environment()
    runs: List[Dict[str, Any]] = []
    git_sha = _get_git_sha()

    print(f"PR79 Python Reference Runner — SHA: {git_sha}")
    print()

    # --- Linear via statsmodels ---
    print("=== Linear (statsmodels) ===")
    X, y, _ = generate_linear_full_rank()
    cp = case_params_linear()
    case_id = make_case_id(cp)
    for cov in ["nonrobust", "hc0", "hc1"]:
        mc = {"model_id": "LinearRegression", "framework": "statsmodels",
              "cov_type": cov}
        result, err = safe_run(ref_linear_statsmodels, X, y, cov_type=cov)
        if err:
            print(f"  statsmodels {cov}: FAILED — {err}")
            continue
        runs.append(make_raw_run(
            f"ref-linear-sm-{cov}", case_id, make_method_config_id(mc),
            "LinearRegression", "statsmodels", "numpy", mc,
            {}, result, status="success" if not err else "failed", error=err,
        ))
        print(f"  statsmodels {cov}: coef={result['coef_'][:2]}...")

    # --- Ridge via sklearn ---
    print("=== Ridge (sklearn) ===")
    X, y, _ = generate_linear_full_rank(200, 8, seed=43)
    cp_ridge = {"domain": "linear", "n_samples": 200, "n_features": 8, "seed": 43}
    case_id = make_case_id(cp_ridge)
    for alpha in [0.1, 1.0, 10.0]:
        mc = {"model_id": "Ridge", "framework": "sklearn", "alpha": alpha}
        result, err = safe_run(ref_ridge_sklearn, X, y, alpha=alpha)
        if err:
            print(f"  sklearn Ridge alpha={alpha}: FAILED — {err}")
            continue
        runs.append(make_raw_run(
            f"ref-ridge-sk-{alpha}", case_id, make_method_config_id(mc),
            "Ridge", "sklearn", "numpy", mc, {}, result,
        ))
        print(f"  sklearn Ridge alpha={alpha}: intercept={result['intercept_']:.4f}")

    # --- CoxPH via statsmodels ---
    print("=== CoxPH (statsmodels) ===")
    X, time_, event, _ = generate_coxph_no_ties()
    cp = case_params_coxph_no_ties()
    case_id = make_case_id(cp)
    mc = {"model_id": "CoxPH", "framework": "statsmodels", "ties": "efron"}
    result, err = safe_run(ref_coxph_statsmodels, X, time_, event, ties="efron")
    if err:
        print(f"  statsmodels CoxPH: FAILED — {err}")
    else:
        runs.append(make_raw_run(
            f"ref-cox-sm", case_id, make_method_config_id(mc),
            "CoxPH", "statsmodels", "numpy", mc, {}, result,
        ))
        print(f"  statsmodels CoxPH: coef={result['coef_'][:2]}..., ll={result['_log_likelihood']:.4f}")

    print(f"\nTotal reference runs: {len(runs)}")
    output_path = "results/pr79/smoke/reference_benchmark.json"
    out = {
        "source_schema_version": "pr79-benchmark-source-1.0",
        "benchmark_session_id": f"pr79-{git_sha[:7]}-references",
        "git_sha": git_sha,
        "environment": env,
        "runs": runs,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {output_path}")
    return runs


if __name__ == "__main__":
    run_all()
