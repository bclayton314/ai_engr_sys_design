import argparse
import json

from config import load_config
from train import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run an ML model benchmarking experiment."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the benchmark config JSON file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    result = run_benchmark(config)

    print("\n=== Benchmark Complete ===")
    print(f"Experiment: {result['experiment_name']}")
    print(f"Ranking metric: {result['ranking_metric']}")
    print("\n=== Leaderboard ===")

    print_leaderboard(result["leaderboard"])


def print_leaderboard(leaderboard: list[dict]) -> None:
    """
    Print a simple terminal leaderboard.
    """
    header = (
        f"{'Rank':<6}"
        f"{'Model':<32}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'ROC-AUC':<12}"
    )

    print(header)
    print("-" * len(header))

    for row in leaderboard:
        print(
            f"{row['rank']:<6}"
            f"{row['model_name']:<32}"
            f"{row['accuracy']:<12.4f}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}"
            f"{row['f1']:<12.4f}"
            f"{row['roc_auc']:<12.4f}"
        )


if __name__ == "__main__":
    main()