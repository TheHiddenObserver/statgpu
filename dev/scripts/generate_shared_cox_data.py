import argparse
import os

import numpy as np


def generate(n: int, p: int, seed: int, tie_digits: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.35, size=p)
    lin = X @ beta
    base = 0.03
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    t_true = -np.log(u) / (base * np.exp(np.clip(lin, -20, 20)))
    censor = rng.exponential(scale=np.median(t_true), size=n)
    event = (t_true <= censor).astype(np.int32)
    time_obs = np.round(np.minimum(t_true, censor), tie_digits)
    return X, time_obs, event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tie-digits", type=int, default=2)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    X, time_obs, event = generate(args.n, args.p, args.seed, args.tie_digits)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        cols = ["time", "event"] + [f"x{i+1}" for i in range(args.p)]
        f.write(",".join(cols) + "\n")
        for i in range(args.n):
            row = [time_obs[i], int(event[i])] + X[i].tolist()
            f.write(",".join(str(v) for v in row) + "\n")

    print(f"wrote {args.out} n={args.n} p={args.p} events={int(event.sum())}")


if __name__ == "__main__":
    main()
