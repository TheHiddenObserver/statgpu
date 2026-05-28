"""Command line entry point for statgpu's automatic analysis agent."""

from __future__ import annotations

import argparse

from ._analysis import StatGPUAnalysisAgent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run statgpu automatic analysis on a CSV file.")
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--target", help="Target column for supervised analysis")
    parser.add_argument("--time", help="Survival time column")
    parser.add_argument("--event", help="Survival event column")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "regression", "classification", "binary", "poisson", "survival", "unsupervised"],
        help="Analysis task. Defaults to automatic inference.",
    )
    parser.add_argument("--device", default="auto", help="statgpu device: auto, cpu, cuda, or torch")
    parser.add_argument("--output", help="Optional markdown report path")
    parser.add_argument("--output-json", help="Optional JSON artifact path")
    parser.add_argument("--output-notebook", help="Optional Jupyter notebook path")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (0 to disable)")
    parser.add_argument(
        "--multiple-testing",
        default="none",
        choices=["none", "bh", "by", "holm", "bonferroni", "hochberg"],
        help="Multiple testing correction method. Default: none (no correction).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for multiple testing")
    args = parser.parse_args(argv)

    agent = StatGPUAnalysisAgent(
        device=args.device,
        cv_folds=args.cv,
        multiple_testing_method=args.multiple_testing,
        alpha=args.alpha,
    )
    result = agent.analyze_csv(
        args.csv,
        target=args.target,
        task=args.task,
        time=args.time,
        event=args.event,
    )

    if args.output:
        result.save_markdown(args.output)
    else:
        print(result.to_markdown())

    if args.output_json:
        result.save_json(args.output_json)

    if args.output_notebook:
        result.save_notebook(args.csv, args.output_notebook)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
