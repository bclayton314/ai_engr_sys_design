import json
from datetime import datetime, timezone
from pathlib import Path

from data import create_dataset, split_dataset
from metrics import compute_evaluation_report
from models import build_model
from tracker import RunTracker
from utils import ensure_dir
from validation import run_cross_validation, flatten_cv_metrics


def run_benchmark(config: dict) -> dict:
    """
    Run all models defined in the config and produce a ranked leaderboard.
    """
    experiment_name = config["experiment_name"]
    random_seed = config["random_seed"]
    ranking_metric = config.get("ranking_metric", "f1")

    X, y = create_dataset(
        dataset_config=config["dataset"],
        random_seed=random_seed,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X=X,
        y=y,
        dataset_config=config["dataset"],
        random_seed=random_seed,
    )

    run_results = []

    for model_config in config["models"]:
        result = run_single_model(
            experiment_name=experiment_name,
            full_config=config,
            model_config=model_config,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        run_results.append(result)

    leaderboard = build_leaderboard(
        run_results=run_results,
        ranking_metric=ranking_metric,
    )

    leaderboard_path = save_leaderboard(
        experiment_name=experiment_name,
        leaderboard=leaderboard,
    )

    return {
        "experiment_name": experiment_name,
        "ranking_metric": ranking_metric,
        "leaderboard_path": leaderboard_path,
        "leaderboard": leaderboard,
    }


def run_single_model(
    experiment_name: str,
    full_config: dict,
    model_config: dict,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict:
    """
    Train, evaluate, optionally cross-validate, and save artifacts for one model.
    """
    model_name = model_config["name"]

    tracker = RunTracker(
        experiment_name=experiment_name,
        model_name=model_name,
    )

    run_config = {
        "experiment_name": experiment_name,
        "random_seed": full_config["random_seed"],
        "dataset": full_config["dataset"],
        "cross_validation": full_config.get("cross_validation", {"enabled": False}),
        "model": model_config,
    }

    tracker.save_config(run_config)

    model = build_model(model_config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model {model_name} does not support predict_proba, "
            "which is required for ROC-AUC."
        )

    y_prob = model.predict_proba(X_test)[:, 1]

    evaluation_report = compute_evaluation_report(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    holdout_metrics = evaluation_report["summary_metrics"]

    tracker.save_metrics(holdout_metrics)
    tracker.save_evaluation_report(evaluation_report)

    cv_report = None
    cv_metrics_flat = {}

    cv_config = full_config.get("cross_validation", {"enabled": False})

    if cv_config.get("enabled", False):
        cv_report = run_cross_validation(
            model=model,
            X=X,
            y=y,
            n_splits=cv_config.get("n_splits", 5),
            shuffle=cv_config.get("shuffle", True),
            random_seed=full_config["random_seed"],
        )

        tracker.save_cross_validation_report(cv_report)
        cv_metrics_flat = flatten_cv_metrics(cv_report)

    model_path = tracker.save_model(model)

    metadata = {
        "run_id": tracker.run_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "model_type": model_config["type"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cross_validation_enabled": bool(cv_config.get("enabled", False)),
        "model_artifact_path": model_path,
        "saved_files": {
            "config": str(tracker.run_dir / "config.json"),
            "metrics": str(tracker.run_dir / "metrics.json"),
            "evaluation_report": str(tracker.run_dir / "evaluation_report.json"),
            "cross_validation_report": (
                str(tracker.run_dir / "cross_validation_report.json")
                if cv_report is not None
                else None
            ),
            "metadata": str(tracker.run_dir / "metadata.json"),
            "model": model_path,
        },
    }

    tracker.save_metadata(metadata)

    all_metrics = {
        **holdout_metrics,
        **cv_metrics_flat,
    }

    return {
        "run_id": tracker.run_id,
        "model_name": model_name,
        "model_type": model_config["type"],
        "metrics": all_metrics,
        "evaluation_report_path": str(tracker.run_dir / "evaluation_report.json"),
        "cross_validation_report_path": (
            str(tracker.run_dir / "cross_validation_report.json")
            if cv_report is not None
            else None
        ),
        "model_path": model_path,
        "run_dir": str(tracker.run_dir),
    }


def build_leaderboard(run_results: list[dict], ranking_metric: str) -> list[dict]:
    """
    Build a sorted leaderboard from all model run results.
    """
    for result in run_results:
        if ranking_metric not in result["metrics"]:
            raise ValueError(
                f"Ranking metric '{ranking_metric}' not found in metrics. "
                f"Available metrics: {list(result['metrics'].keys())}"
            )

    leaderboard = []

    for result in run_results:
        row = {
            "run_id": result["run_id"],
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "model_path": result["model_path"],
            "run_dir": result["run_dir"],
            "evaluation_report_path": result["evaluation_report_path"],
            "cross_validation_report_path": result["cross_validation_report_path"],
            **result["metrics"],
        }
        leaderboard.append(row)

    leaderboard.sort(
        key=lambda row: row[ranking_metric],
        reverse=True,
    )

    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    return leaderboard


def save_leaderboard(experiment_name: str, leaderboard: list[dict]) -> str:
    """
    Save the leaderboard under runs/leaderboards.
    """
    leaderboard_dir = ensure_dir(Path("runs") / "leaderboards")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = leaderboard_dir / f"{experiment_name}_{timestamp}_leaderboard.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    return str(path)