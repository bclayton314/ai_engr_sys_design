import os

import pytest

from train import build_leaderboard, run_benchmark


def test_build_leaderboard_sorts_by_ranking_metric():
    run_results = [
        {
            "run_id": "run_1",
            "model_name": "model_a",
            "model_type": "logistic_regression",
            "model_path": "path_a",
            "evaluation_report_path": "report_a",
            "cross_validation_report_path": "cv_report_a",
            "run_dir": "dir_a",
            "metrics": {
                "accuracy": 0.80,
                "precision": 0.81,
                "recall": 0.79,
                "f1": 0.80,
                "roc_auc": 0.85,
                "cv_f1_mean": 0.79,
                "cv_f1_std": 0.02,
            },
        },
        {
            "run_id": "run_2",
            "model_name": "model_b",
            "model_type": "random_forest",
            "model_path": "path_b",
            "evaluation_report_path": "report_b",
            "cross_validation_report_path": "cv_report_b",
            "run_dir": "dir_b",
            "metrics": {
                "accuracy": 0.90,
                "precision": 0.91,
                "recall": 0.89,
                "f1": 0.90,
                "roc_auc": 0.95,
                "cv_f1_mean": 0.88,
                "cv_f1_std": 0.01,
            },
        },
    ]

    leaderboard = build_leaderboard(run_results, ranking_metric="cv_f1_mean")

    assert leaderboard[0]["model_name"] == "model_b"
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[0]["evaluation_report_path"] == "report_b"
    assert leaderboard[0]["cross_validation_report_path"] == "cv_report_b"
    assert leaderboard[1]["model_name"] == "model_a"
    assert leaderboard[1]["rank"] == 2


def test_build_leaderboard_invalid_ranking_metric():
    run_results = [
        {
            "run_id": "run_1",
            "model_name": "model_a",
            "model_type": "logistic_regression",
            "model_path": "path_a",
            "evaluation_report_path": "report_a",
            "cross_validation_report_path": None,
            "run_dir": "dir_a",
            "metrics": {
                "accuracy": 0.80,
            },
        }
    ]

    with pytest.raises(ValueError, match="Ranking metric"):
        build_leaderboard(run_results, ranking_metric="f1")


def test_run_benchmark_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = {
        "experiment_name": "test_benchmark",
        "random_seed": 42,
        "ranking_metric": "cv_f1_mean",
        "dataset": {
            "n_samples": 120,
            "n_features": 10,
            "n_informative": 5,
            "n_redundant": 2,
            "n_classes": 2,
            "test_size": 0.2,
        },
        "cross_validation": {
            "enabled": True,
            "n_splits": 3,
            "shuffle": True,
        },
        "models": [
            {
                "name": "logistic_regression_test",
                "type": "logistic_regression",
                "params": {
                    "max_iter": 100,
                    "solver": "liblinear",
                },
            },
            {
                "name": "decision_tree_test",
                "type": "decision_tree",
                "params": {
                    "max_depth": 3,
                    "random_state": 42,
                },
            },
        ],
    }

    result = run_benchmark(config)

    assert result["experiment_name"] == "test_benchmark"
    assert result["ranking_metric"] == "cv_f1_mean"
    assert len(result["leaderboard"]) == 2
    assert os.path.exists(result["leaderboard_path"])

    assert os.path.exists("runs")
    assert os.path.exists("artifacts")
    assert os.path.exists("runs/leaderboards")

    for row in result["leaderboard"]:
        assert "rank" in row
        assert "model_name" in row
        assert "accuracy" in row
        assert "precision" in row
        assert "recall" in row
        assert "f1" in row
        assert "roc_auc" in row
        assert "cv_f1_mean" in row
        assert "cv_f1_std" in row
        assert "evaluation_report_path" in row
        assert "cross_validation_report_path" in row
        assert os.path.exists(row["evaluation_report_path"])
        assert os.path.exists(row["cross_validation_report_path"])