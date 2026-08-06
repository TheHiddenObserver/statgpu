"""
Dedicated benchmark for CoxPH covariance modes:
  - nonrobust
  - hc1
  - cluster

Compares:
  - statgpu CPU
  - statgpu GPU (if available)
  - statsmodels PHReg (if available)
  - R survival::coxph (if available)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.survival import CoxPH
from statgpu._config import set_device, cuda_available

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    import statsmodels.duration.api as smd
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Cox cov_type modes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--p", type=int, default=10)
    p.add_argument(
        "--ties",
        type=str,
        default="breslow",
        choices=["breslow", "efron"],
    )
    p.add_argument("--groups", type=int, default=120)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--json-out", type=str, default="")
    return p.parse_args()


def make_data(seed: int, n: int, p: int, groups: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.35, size=p)
    lin = X @ beta
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    t_true = -np.log(u) / (0.03 * np.exp(np.clip(lin, -20, 20)))
    censor = rng.exponential(scale=np.median(t_true), size=n)
    event = (t_true <= censor).astype(int)
    t_obs = np.minimum(t_true, censor)
    cluster = rng.integers(0, max(2, groups), size=n)
    return X, t_obs, event, cluster


def time_fit(model: CoxPH, X, t_obs, event, cluster=None):
    t0 = time.perf_counter()
    if cluster is None:
        model.fit(X, t_obs, event)
    else:
        model.fit(X, t_obs, event, cluster=cluster)
    if HAS_CUPY and hasattr(X, "device"):
        cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def safe_diff(a, b):
    if a is None or b is None:
        return np.nan
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            "comparison vectors must have identical shapes, "
            f"got {a.shape} and {b.shape}"
        )
    if a.size == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("comparison vectors must be non-empty and finite")
    return float(np.max(np.abs(a - b)))


def json_ready(value):
    """Convert benchmark results to strict, portable JSON values."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def statsmodels_covariance_capability(cov_type: str) -> Dict[str, Any]:
    """Describe PHReg covariance support without mislabelling model-based SEs."""
    if cov_type == "hc1":
        return {
            "supported": False,
            "contract": "unsupported",
            "reason": (
                "PHReg.fit() does not expose the HC1 "
                "n_units/(n_units-p) score-sandwich contract"
            ),
        }
    return {
        "supported": True,
        "contract": (
            "cluster-aggregated score sandwich without HC1 correction"
            if cov_type == "cluster"
            else "model-based observed-information inverse"
        ),
        "reason": "",
    }


def external_covariance_contract_fields(
    *,
    supported: bool,
    requested_contract: str,
    actual_contract: Optional[str] = None,
    unsupported_reason: str = "",
) -> Dict[str, Any]:
    """Return unambiguous machine-readable external support metadata."""
    supported = bool(supported)
    reason = str(unsupported_reason).strip()
    if not supported and not reason:
        reason = "external covariance result is unavailable"
    return {
        "covariance_contract": (
            str(actual_contract or requested_contract)
            if supported
            else "unsupported"
        ),
        "requested_covariance_contract": str(requested_contract),
        "unsupported_reason": "" if supported else reason,
    }


def validate_external_vector(value, n_features: int, *, name: str):
    """Return a finite external vector with the exact expected length."""
    if value is None:
        raise ValueError(f"{name} is missing")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(n_features):
        raise ValueError(
            f"{name} must contain exactly {int(n_features)} values, got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def statsmodels_result_has_finite_inference(result, n_features: int) -> bool:
    """Return whether PHReg produced complete, finite coefficient inference."""
    for attribute in ("params", "bse", "pvalues"):
        try:
            validate_external_vector(
                getattr(result, attribute, None),
                n_features,
                name=f"statsmodels {attribute}",
            )
        except ValueError:
            return False
    return True


def run_r(
    csv_path: Path,
    ties: str,
    cov_type: str,
    *,
    n_features: int,
    max_iter: int,
    tol: float,
) -> Dict[str, Any]:
    """Run a precisely labelled R survival covariance reference."""
    if shutil.which("Rscript") is None:
        return {"supported": False, "error": "Rscript not found"}
    if cov_type not in {"nonrobust", "hc1", "cluster"}:
        return {"supported": False, "error": f"unsupported cov_type={cov_type}"}

    robust = "TRUE" if cov_type in {"hc1", "cluster"} else "FALSE"
    add_cluster = cov_type == "cluster"
    apply_hc1 = cov_type == "hc1"
    r_script = f"""
    suppressPackageStartupMessages(library(survival))
    tryCatch({{
      d <- read.csv("{csv_path.as_posix()}")
      feature_names <- grep("^x[0-9]+$", names(d), value=TRUE)
      p <- length(feature_names)
      rhs <- paste(feature_names, collapse=" + ")
      if ({str(add_cluster).upper()}) rhs <- paste(rhs, "+ cluster(cluster)")
      form <- as.formula(paste("Surv(time, event) ~", rhs))
      n_units <- if ({str(add_cluster).upper()}) length(unique(d$cluster)) else nrow(d)
      if ({str(apply_hc1).upper()} && n_units <= p) stop("HC1 requires n_units > p")
      started <- proc.time()[["elapsed"]]
      fit <- coxph(
        form,
        data=d,
        ties="{ties}",
        robust={robust},
        singular.ok=FALSE,
        control=coxph.control(
          iter.max={int(max_iter)},
          eps={float(tol)!r},
          timefix=FALSE
        )
      )
      fit_ms <- (proc.time()[["elapsed"]] - started) * 1000
      covariance <- fit$var
      correction <- 1.0
      if ({str(apply_hc1).upper()}) {{
        correction <- n_units / (n_units - p)
        covariance <- correction * covariance
      }}
      coef <- stats::coef(fit)
      bse <- sqrt(diag(covariance))
      pvalues <- 2 * pnorm(-abs(coef / bse))
      cat("FIT_MS=", format(fit_ms, digits=17), "\n", sep="")
      cat("N_UNITS=", n_units, "\n", sep="")
      cat("CORRECTION=", format(correction, digits=17), "\n", sep="")
      cat("COEF=", paste(format(coef, digits=17, scientific=TRUE), collapse=","), "\n", sep="")
      cat("BSE=", paste(format(bse, digits=17, scientific=TRUE), collapse=","), "\n", sep="")
      cat("PVALUES=", paste(format(pvalues, digits=17, scientific=TRUE), collapse=","), "\n", sep="")
    }}, error=function(exc) {{
      message(conditionMessage(exc))
      quit(status=2)
    }})
    """
    try:
        completed = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"supported": False, "error": f"Rscript failed: {exc}"}
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return {
            "supported": False,
            "error": f"R survival::coxph failed: {message}",
        }
    fields = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    required = {"FIT_MS", "N_UNITS", "CORRECTION", "COEF", "BSE", "PVALUES"}
    if not required.issubset(fields):
        return {
            "supported": False,
            "error": f"R output missing fields: {sorted(required - fields.keys())}",
        }
    try:
        fit_ms = float(fields["FIT_MS"])
        n_units = int(fields["N_UNITS"])
        correction = float(fields["CORRECTION"])
        vectors = {
            name: validate_external_vector(
                np.fromstring(fields[field], sep=","),
                n_features,
                name=f"R {name}",
            )
            for name, field in (
                ("coef", "COEF"),
                ("bse", "BSE"),
                ("pvalues", "PVALUES"),
            )
        }
        if (
            not np.isfinite(fit_ms)
            or fit_ms < 0.0
            or n_units < 1
            or not np.isfinite(correction)
            or correction <= 0.0
        ):
            raise ValueError("R scalar diagnostics are invalid")
    except (TypeError, ValueError) as exc:
        return {"supported": False, "error": f"invalid R output: {exc}"}
    return {
        "supported": True,
        "fit_ms": fit_ms,
        "n_units": n_units,
        "correction": correction,
        **vectors,
    }


def main():
    args = parse_args()
    X, t_obs, event, cluster = make_data(args.seed, args.n, args.p, args.groups)
    temporary_directory = tempfile.TemporaryDirectory(prefix="statgpu-cox-cluster-")
    csv_path = Path(temporary_directory.name) / "cox_cluster.csv"
    header = ["time", "event", "cluster"] + [
        f"x{index + 1}" for index in range(args.p)
    ]
    np.savetxt(
        csv_path,
        np.column_stack((t_obs, event, cluster, X)),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    if HAS_CUPY and cuda_available():
        # Warm up CUDA context and cuBLAS handles outside timing.
        _ = cp.asarray([1.0, 2.0]) @ cp.asarray([3.0, 4.0])
        cp.cuda.Stream.null.synchronize()

    rows = []
    for cov in ["nonrobust", "hc1", "cluster"]:
        n_units = (
            int(np.unique(cluster).size) if cov == "cluster" else int(args.n)
        )
        correction = (
            n_units / (n_units - args.p) if cov == "hc1" else 1.0
        )
        covariance_contract = {
            "nonrobust": "model-based observed-information inverse",
            "hc1": "row-score sandwich times n_units/(n_units-p)",
            "cluster": "cluster-aggregated score sandwich without HC1 correction",
        }[cov]
        # statgpu CPU
        set_device("cpu")
        m_cpu = CoxPH(
            device="cpu",
            ties=args.ties,
            cov_type=cov,
            max_iter=args.max_iter,
            tol=args.tol,
            compute_inference=True,
        )
        ms_cpu = time_fit(
            m_cpu,
            X,
            t_obs,
            event,
            cluster if cov == "cluster" else None,
        )
        rows.append(
            {
                "method": "CoxPH",
                "framework": f"statgpu-cpu({cov})",
                "fit_ms": ms_cpu,
                "coef_ref_diff": 0.0,
                "bse_ref_diff": 0.0,
                "p_ref_diff": 0.0,
                "supported": True,
                "independent_units": n_units if cov != "nonrobust" else None,
                "finite_sample_correction": correction,
                "covariance_contract": covariance_contract,
                "notes": "reference for this covariance mode",
            }
        )

        # statgpu GPU
        if HAS_CUPY and cuda_available():
            set_device("cuda")
            Xg = cp.asarray(X)
            tg = cp.asarray(t_obs)
            eg = cp.asarray(event)
            cg = cp.asarray(cluster)
            m_gpu = CoxPH(
                device="cuda",
                ties=args.ties,
                cov_type=cov,
                max_iter=args.max_iter,
                tol=args.tol,
                compute_inference=True,
            )
            ms_gpu = time_fit(m_gpu, Xg, tg, eg, cg if cov == "cluster" else None)
            rows.append(
                {
                    "method": "CoxPH",
                    "framework": f"statgpu-gpu({cov})",
                    "fit_ms": ms_gpu,
                    "coef_ref_diff": safe_diff(m_cpu.coef_, m_gpu.coef_),
                    "bse_ref_diff": safe_diff(m_cpu._bse, m_gpu._bse),
                    "p_ref_diff": safe_diff(m_cpu._pvalues, m_gpu._pvalues),
                    "supported": True,
                    "independent_units": n_units if cov != "nonrobust" else None,
                    "finite_sample_correction": correction,
                    "covariance_contract": covariance_contract,
                    "notes": "ref=statgpu-cpu",
                }
            )

        # statsmodels
        if HAS_STATSMODELS:
            sm_capability = statsmodels_covariance_capability(cov)
            if not sm_capability["supported"]:
                rows.append(
                    {
                        "method": "CoxPH",
                        "framework": "statsmodels.PHReg(hc1)",
                        "fit_ms": np.nan,
                        "coef_ref_diff": np.nan,
                        "bse_ref_diff": np.nan,
                        "p_ref_diff": np.nan,
                        "supported": False,
                        "independent_units": n_units,
                        "finite_sample_correction": correction,
                        **external_covariance_contract_fields(
                            supported=False,
                            requested_contract=covariance_contract,
                            unsupported_reason=sm_capability["reason"],
                        ),
                        "notes": f"unsupported: {sm_capability['reason']}",
                    }
                )
                sm_result_supported = False
            else:
                sm_result_supported = True
            try:
                if sm_result_supported:
                    t0 = time.perf_counter()
                    sm_model = smd.PHReg(t_obs, X, status=event, ties=args.ties)
                    sm_res = (
                        sm_model.fit(
                            groups=cluster,
                            method="newton",
                            maxiter=args.max_iter,
                            tol=args.tol,
                            disp=False,
                        )
                        if cov == "cluster"
                        else sm_model.fit(
                            method="newton",
                            maxiter=args.max_iter,
                            tol=args.tol,
                            disp=False,
                        )
                    )
                    t1 = time.perf_counter()
                    finite_inference = statsmodels_result_has_finite_inference(
                        sm_res, args.p
                    )
                    rows.append(
                        {
                            "method": "CoxPH",
                            "framework": f"statsmodels.PHReg({cov})",
                            "fit_ms": (t1 - t0) * 1000.0,
                            "coef_ref_diff": (
                                safe_diff(m_cpu.coef_, sm_res.params)
                                if finite_inference
                                else np.nan
                            ),
                            "bse_ref_diff": (
                                safe_diff(m_cpu._bse, sm_res.bse)
                                if finite_inference
                                else np.nan
                            ),
                            "p_ref_diff": (
                                safe_diff(m_cpu._pvalues, sm_res.pvalues)
                                if finite_inference
                                else np.nan
                            ),
                            "supported": finite_inference,
                            "independent_units": (
                                n_units if cov == "cluster" else None
                            ),
                            "finite_sample_correction": 1.0,
                            **external_covariance_contract_fields(
                                supported=finite_inference,
                                requested_contract=covariance_contract,
                                actual_contract=sm_capability["contract"],
                                unsupported_reason=(
                                    "PHReg returned non-finite coefficient "
                                    "inference"
                                ),
                            ),
                            "notes": (
                                "ref=statgpu-cpu"
                                if finite_inference
                                else (
                                    "unsupported: PHReg returned non-finite "
                                    "coefficient inference"
                                )
                            ),
                        }
                    )
            except Exception as e:
                rows.append(
                    {
                        "method": "CoxPH",
                        "framework": f"statsmodels.PHReg({cov})",
                        "fit_ms": np.nan,
                        "coef_ref_diff": np.nan,
                        "bse_ref_diff": np.nan,
                        "p_ref_diff": np.nan,
                        "supported": False,
                        "independent_units": (
                            n_units if cov != "nonrobust" else None
                        ),
                        "finite_sample_correction": correction,
                        **external_covariance_contract_fields(
                            supported=False,
                            requested_contract=covariance_contract,
                            unsupported_reason=f"{type(e).__name__}: {e}",
                        ),
                        "notes": f"skipped: {e}",
                    }
                )

        r_result = run_r(
            csv_path,
            args.ties,
            cov,
            n_features=args.p,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        r_label = {
            "nonrobust": "R survival::coxph(nonrobust)",
            "hc1": "R survival::coxph(robust-score + explicit HC1 correction)",
            "cluster": "R survival::coxph(cluster-robust)",
        }[cov]
        r_supported = bool(r_result.get("supported", False))
        r_unsupported_reason = r_result.get("error", "")
        rows.append(
            {
                "method": "CoxPH",
                "framework": r_label,
                "fit_ms": r_result.get("fit_ms", np.nan),
                "coef_ref_diff": safe_diff(m_cpu.coef_, r_result.get("coef")),
                "bse_ref_diff": safe_diff(m_cpu._bse, r_result.get("bse")),
                "p_ref_diff": safe_diff(m_cpu._pvalues, r_result.get("pvalues")),
                "supported": r_supported,
                "independent_units": r_result.get("n_units", n_units),
                "finite_sample_correction": r_result.get("correction", correction),
                **external_covariance_contract_fields(
                    supported=r_supported,
                    requested_contract=covariance_contract,
                    actual_contract=covariance_contract,
                    unsupported_reason=r_unsupported_reason,
                ),
                "notes": (
                    "ref=statgpu-cpu; "
                    + (
                        "R robust score sandwich with explicit n_units/(n_units-p) correction"
                        if cov == "hc1" and r_result.get("supported")
                        else r_result.get("error", "native R covariance mode")
                    )
                ),
            }
        )

    solver_controls = {
        "ties": args.ties,
        "solver": "newton",
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    for row in rows:
        row["solver_controls"] = dict(solver_controls)

    print("\n=== Cox Covariance Benchmark ===")
    print(
        f"{'framework':<34} {'fit_ms':>10} {'coef_diff':>12} "
        f"{'bse_diff':>12} {'p_diff':>12}"
    )
    for r in rows:
        print(
            f"{r['framework']:<34} {r['fit_ms']:>10.2f} "
            f"{r['coef_ref_diff']:>12.3e} {r['bse_ref_diff']:>12.3e} {r['p_ref_diff']:>12.3e}"
        )
        if r["notes"]:
            print(f"  note: {r['notes']}")

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(json_ready(rows), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON: {out}")
    temporary_directory.cleanup()


if __name__ == "__main__":
    main()
