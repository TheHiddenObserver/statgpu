import json
import os
import subprocess
import tempfile

import numpy as np

from statgpu._config import set_device
from statgpu.survival import CoxPH


def make_data(n=800, p=7, seed=77):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.35, size=p)
    lin = X @ beta
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    base = 0.03
    t_true = -np.log(u) / (base * np.exp(np.clip(lin, -20, 20)))
    censor = rng.exponential(scale=np.median(t_true), size=n)
    event = (t_true <= censor).astype(np.int32)
    time_obs = np.minimum(t_true, censor)
    # enforce ties
    time_obs = np.round(time_obs, 2)
    if np.sum(event) > 0:
        ev_t = time_obs[event == 1]
        uft, cnt = np.unique(ev_t, return_counts=True)
        if len(uft) == 0 or cnt.max() < 2:
            tt = np.unique(time_obs)[0]
            ix = np.where(time_obs == tt)[0]
            event[:] = 0
            event[ix[: min(6, len(ix))]] = 1
    return X.astype(float), time_obs.astype(float), event.astype(int)


def main():
    X, time_obs, event = make_data()

    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            cols = ["time", "event"] + [f"x{i+1}" for i in range(X.shape[1])]
            f.write(",".join(cols) + "\n")
            for i in range(X.shape[0]):
                row = [time_obs[i], int(event[i])] + X[i].tolist()
                f.write(",".join(str(v) for v in row) + "\n")

        set_device("cpu")
        all_ok = True
        for ties in ["breslow", "efron"]:
            m = CoxPH(
                ties=ties,
                device="cpu",
                compute_inference=True,
                max_iter=80,
                tol=1e-8,
            )
            m.fit(X, time_obs, event)

            py = {
                "coef": m.coef_.tolist(),
                "bse": m._bse.tolist(),
                "ll": float(m._log_likelihood),
            }

            r_code = f"""
              suppressPackageStartupMessages(library(survival))
              suppressPackageStartupMessages(library(jsonlite))
              d <- read.csv("{csv_path}")
              vars <- paste0("x", 1:{X.shape[1]})
              form <- as.formula(paste("Surv(time, event) ~", paste(vars, collapse=" + ")))
              fit <- coxph(form, data=d, ties="{ties}")
              out <- list(coef=as.numeric(coef(fit)), bse=as.numeric(sqrt(diag(vcov(fit)))), ll=as.numeric(fit$loglik[2]))
              cat(jsonlite::toJSON(out, auto_unbox=TRUE))
            """
            proc = subprocess.run(
                ["Rscript", "-e", r_code], text=True, capture_output=True
            )
            if proc.returncode != 0:
                print(f"[FAIL] R run failed for ties={ties}")
                print(proc.stderr.strip())
                all_ok = False
                continue

            txt = proc.stdout.strip().splitlines()
            js = txt[-1] if txt else ""
            r = json.loads(js)

            coef_close = np.allclose(
                np.array(py["coef"]), np.array(r["coef"]), rtol=2e-2, atol=2e-3
            )
            bse_close = np.allclose(
                np.array(py["bse"]), np.array(r["bse"]), rtol=2e-1, atol=2e-3
            )
            ll_close = np.allclose(py["ll"], float(r["ll"]), rtol=2e-2, atol=2e-3)
            ok = bool(coef_close and bse_close and ll_close)
            all_ok = all_ok and ok
            label = "PASS" if ok else "FAIL"
            print(
                f"[{label}] R-compare-{ties} coef={coef_close} bse={bse_close} ll={ll_close}"
            )

        if not all_ok:
            raise SystemExit(1)
        print("SUMMARY: R comparison passed")
    finally:
        try:
            os.unlink(csv_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
