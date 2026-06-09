"""
Precision and performance regression test for PR #54 review fixes.

Compares CV results before/after changes on the remote GPU server.
Run BEFORE making changes to generate baseline, then AFTER to verify.

Usage:
    # Generate baseline (run before changes):
    python dev/tests/_precision_regression.py --mode=baseline

    # Verify against baseline (run after changes):
    python dev/tests/_precision_regression.py --mode=verify

    # Performance benchmark only:
    python dev/tests/_precision_regression.py --mode=performance
"""
import json
import os
import sys
import time

import numpy as np

# Test configurations: (loss, penalty, n_samples, n_features, l1_ratio, description)
TEST_CONFIGS = [
    ("squared_error", "l1", 500, 50, 1.0, "SE+L1 basic"),
    ("squared_error", "elasticnet", 500, 50, 0.5, "SE+EN basic"),
    ("squared_error", "scad", 500, 50, 1.0, "SE+SCAD"),
    ("squared_error", "mcp", 500, 50, 1.0, "SE+MCP"),
    ("logistic", "l1", 500, 50, 1.0, "Logistic+L1"),
    ("logistic", "elasticnet", 500, 50, 0.5, "Logistic+EN"),
    ("logistic", "scad", 500, 50, 1.0, "Logistic+SCAD"),
    ("poisson", "l1", 500, 50, 1.0, "Poisson+L1"),
    ("gamma", "l1", 500, 50, 1.0, "Gamma+L1"),
    ("tweedie", "l1", 500, 50, 1.0, "Tweedie+L1"),
]

PERF_CONFIGS = [
    ("squared_error", "l1", 5000, 200, 1.0, "SE+L1 medium"),
    ("squared_error", "scad", 5000, 200, 1.0, "SE+SCAD medium"),
    ("logistic", "l1", 5000, 200, 1.0, "Logistic+L1 medium"),
    ("poisson", "l1", 5000, 200, 1.0, "Poisson+L1 medium"),
]

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "_precision_baseline.json")
TOLERANCES = {
    "alpha_rel": 1e-6,       # relative tolerance for best alpha
    "coef_rel": 1e-4,        # relative tolerance for coefficients
    "score_rel": 1e-5,       # relative tolerance for CV scores
}


def generate_data(loss, n, p, random_state=42):
    """Generate test data appropriate for the loss function."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, p)
    if loss == "logistic":
        coef_true = rng.randn(p) * 0.5
        prob = 1 / (1 + np.exp(-X @ coef_true))
        y = (rng.rand(n) < prob).astype(float)
    elif loss == "poisson":
        coef_true = rng.randn(p) * 0.3
        mu = np.exp(np.clip(X @ coef_true, -5, 5))
        y = rng.poisson(mu).astype(float)
    elif loss in ("gamma", "inverse_gaussian"):
        coef_true = rng.randn(p) * 0.2
        mu = np.exp(np.clip(X @ coef_true, -5, 5))
        y = np.abs(mu) + rng.rand(n) * 0.1
    elif loss == "tweedie":
        coef_true = rng.randn(p) * 0.2
        mu = np.exp(np.clip(X @ coef_true, -5, 5))
        y = mu + rng.randn(n) * 0.1
        y = np.maximum(y, 0.01)
    else:  # squared_error
        coef_true = rng.randn(p)
        y = X @ coef_true + rng.randn(n) * 0.5
    return X, y


def run_cv_and_collect(loss, penalty, n, p, l1_ratio, device="auto"):
    """Run PenalizedGLM_CV and collect key metrics."""
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    X, y = generate_data(loss, n, p)
    model = PenalizedGLM_CV(
        loss=loss,
        penalty=penalty,
        n_alphas=20,
        l1_ratio=l1_ratio,
        cv=3,
        device=device,
        max_iter=500,
        tol=1e-4,
    )
    t0 = time.time()
    model.fit(X, y)
    elapsed = time.time() - t0

    return {
        "alpha": float(model.alpha_),
        "best_score": float(model.best_score_),
        "coef_norm": float(np.linalg.norm(model.coef_)),
        "coef_max": float(np.max(np.abs(model.coef_))),
        "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-6)),
        "mean_scores": model.cv_results_["mean_score"].tolist(),
        "elapsed": elapsed,
    }


def generate_baseline():
    """Generate baseline results and save to JSON."""
    results = {}
    for loss, penalty, n, p, l1_ratio, desc in TEST_CONFIGS:
        print(f"Baseline: {desc} ... ", end="", flush=True)
        try:
            metrics = run_cv_and_collect(loss, penalty, n, p, l1_ratio)
            results[desc] = metrics
            print(f"OK (alpha={metrics['alpha']:.6f}, time={metrics['elapsed']:.2f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results[desc] = {"error": str(e)}

    with open(BASELINE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline saved to {BASELINE_FILE}")


def verify_against_baseline():
    """Run same configs and compare against baseline."""
    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: Baseline file not found: {BASELINE_FILE}")
        print("Run with --mode=baseline first.")
        return False

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    all_ok = True
    for loss, penalty, n, p, l1_ratio, desc in TEST_CONFIGS:
        if desc not in baseline:
            print(f"SKIP: {desc} (no baseline)")
            continue
        if "error" in baseline[desc]:
            print(f"SKIP: {desc} (baseline had error)")
            continue

        print(f"Verify: {desc} ... ", end="", flush=True)
        try:
            current = run_cv_and_collect(loss, penalty, n, p, l1_ratio)
            base = baseline[desc]

            # Check alpha
            alpha_rel = abs(current["alpha"] - base["alpha"]) / max(
                abs(base["alpha"]), 1e-15
            )
            if alpha_rel > TOLERANCES["alpha_rel"]:
                print(
                    f"ALPHA REGRESSION: {base['alpha']:.6f} -> {current['alpha']:.6f} "
                    f"(rel={alpha_rel:.2e})"
                )
                all_ok = False
                continue

            # Check score
            score_rel = abs(current["best_score"] - base["best_score"]) / max(
                abs(base["best_score"]), 1e-15
            )
            if score_rel > TOLERANCES["score_rel"]:
                print(
                    f"SCORE REGRESSION: {base['best_score']:.6f} -> "
                    f"{current['best_score']:.6f} (rel={score_rel:.2e})"
                )
                all_ok = False
                continue

            # Check coef
            coef_rel = abs(current["coef_norm"] - base["coef_norm"]) / max(
                abs(base["coef_norm"]), 1e-15
            )
            if coef_rel > TOLERANCES["coef_rel"]:
                print(
                    f"COEF REGRESSION: norm {base['coef_norm']:.6f} -> "
                    f"{current['coef_norm']:.6f} (rel={coef_rel:.2e})"
                )
                all_ok = False
                continue

            # Check timing (allow 20% regression)
            time_ratio = current["elapsed"] / max(base["elapsed"], 0.01)
            time_flag = f" (SLOWER {time_ratio:.1f}x)" if time_ratio > 1.2 else ""
            print(
                f"OK (alpha={current['alpha']:.6f}, "
                f"time={current['elapsed']:.2f}s{time_flag})"
            )

        except Exception as e:
            print(f"FAILED: {e}")
            all_ok = False

    return all_ok


def run_performance_benchmark():
    """Run performance benchmark and report timing."""
    print("Performance benchmark")
    print("-" * 60)
    results = {}
    for loss, penalty, n, p, l1_ratio, desc in PERF_CONFIGS:
        print(f"Bench: {desc} (n={n}, p={p}) ... ", end="", flush=True)
        try:
            # Run 3 trials, take median
            times = []
            for trial in range(3):
                metrics = run_cv_and_collect(loss, penalty, n, p, l1_ratio)
                times.append(metrics["elapsed"])

            median_time = sorted(times)[1]
            results[desc] = {
                "median_time": median_time,
                "all_times": times,
            }
            print(
                f"{median_time:.2f}s "
                f"(trials: {[f'{t:.2f}' for t in times]})"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            results[desc] = {"error": str(e)}

    print(f"\nResults saved to stdout as JSON")
    print(json.dumps(results, indent=2))
    return True


def main():
    mode = "verify"
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]

    if mode == "baseline":
        generate_baseline()
    elif mode == "verify":
        ok = verify_against_baseline()
        sys.exit(0 if ok else 1)
    elif mode == "performance":
        ok = run_performance_benchmark()
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown mode: {mode}")
        print("Use --mode=baseline, --mode=verify, or --mode=performance")
        sys.exit(1)


if __name__ == "__main__":
    main()
