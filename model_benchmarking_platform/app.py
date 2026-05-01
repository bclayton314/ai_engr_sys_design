import argparse

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
    print(f"Leaderboard saved to: {result['leaderboard_path']}")
    print("\n=== Leaderboard ===")

    print_leaderboard(result["leaderboard"])


def print_leaderboard(leaderboard: list[dict]) -> None:
    """
    Print a terminal leaderboard.

    If cross-validation metrics are available, display them too.
    """
    has_cv = bool(leaderboard) and "cv_f1_mean" in leaderboard[0]

    if has_cv:
        header = (
            f"{'Rank':<6}"
            f"{'Model':<32}"
            f"{'F1':<12}"
            f"{'CV F1 Mean':<14}"
            f"{'CV F1 Std':<12}"
            f"{'ROC-AUC':<12}"
            f"{'CV ROC Mean':<14}"
        )
    else:
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
        if has_cv:
            print(
                f"{row['rank']:<6}"
                f"{row['model_name']:<32}"
                f"{row['f1']:<12.4f}"
                f"{row['cv_f1_mean']:<14.4f}"
                f"{row['cv_f1_std']:<12.4f}"
                f"{row['roc_auc']:<12.4f}"
                f"{row['cv_roc_auc_mean']:<14.4f}"
            )
        else:
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