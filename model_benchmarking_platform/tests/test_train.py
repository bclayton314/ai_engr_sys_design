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
            "run_dir": "dir_a",
            "metrics": {
                "accuracy": 0.80,
                "precision": 0.81,
                "recall": 0.79,
                "f1": 0.80,
                "roc_auc": 0.85,
            },
        },
        {
            "run_id": "run_2",
            "model_name": "model_b",
            "model_type": "random_forest",
            "model_path": "path_b",
            "run_dir": "dir_b",
            "metrics": {
                "accuracy": 0.90,
                "precision": 0.91,
                "recall": 0.89,
                "f1": 0.90,
                "roc_auc": 0.95,
            },
        },
    ]

    leaderboard = build_leaderboard(run_results, ranking_metric="f1")

    assert leaderboard[0]["model_name"] == "model_b"
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["model_name"] == "model_a"
    assert leaderboard[1]["rank"] == 2


def test_build_leaderboard_invalid_ranking_metric():
    run_results = [
        {
            "run_id": "run_1",
            "model_name": "model_a",
            "model_type": "logistic_regression",
            "model_path": "path_a",
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
        "ranking_metric": "f1",
        "dataset": {
            "n_samples": 120,
            "n_features": 10,
            "n_informative": 5,
            "n_redundant": 2,
            "n_classes": 2,
            "test_size": 0.2,
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
    assert result["ranking_metric"] == "f1"
    assert len(result["leaderboard"]) == 2

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