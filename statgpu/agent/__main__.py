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
    args = parser.parse_args(argv)

    agent = StatGPUAnalysisAgent(device=args.device)
    result = agent.analyze_csv(
        args.csv,
        target=args.target,
        task=args.task,
        time=args.time,
        event=args.event,
    )
    markdown = result.to_markdown()
    if args.output:
        result.save_markdown(args.output)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
