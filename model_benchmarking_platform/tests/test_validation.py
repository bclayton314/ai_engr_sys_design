from sklearn.linear_model import LogisticRegression

from data import create_dataset
from validation import (
    run_cross_validation,
    summarize_cross_validation,
    flatten_cv_metrics,
)


def test_run_cross_validation_returns_expected_sections():
    dataset_config = {
        "n_samples": 100,
        "n_features": 10,
        "n_informative": 5,
        "n_redundant": 2,
        "n_classes": 2,
        "test_size": 0.2,
    }

    X, y = create_dataset(dataset_config, random_seed=42)

    model = LogisticRegression(max_iter=1000, solver="liblinear")

    report = run_cross_validation(
        model=model,
        X=X,
        y=y,
        n_splits=3,
        shuffle=True,
        random_seed=42,
    )

    assert "folds" in report
    assert "mean_metrics" in report
    assert "std_metrics" in report
    assert len(report["folds"]) == 3


def test_cross_validation_fold_results_have_metrics():
    dataset_config = {
        "n_samples": 100,
        "n_features": 10,
        "n_informative": 5,
        "n_redundant": 2,
        "n_classes": 2,
        "test_size": 0.2,
    }

    X, y = create_dataset(dataset_config, random_seed=42)

    model = LogisticRegression(max_iter=1000, solver="liblinear")

    report = run_cross_validation(
        model=model,
        X=X,
        y=y,
        n_splits=3,
        shuffle=True,
        random_seed=42,
    )

    first_fold = report["folds"][0]

    assert "fold" in first_fold
    assert "metrics" in first_fold
    assert "train_size" in first_fold
    assert "test_size" in first_fold
    assert "f1" in first_fold["metrics"]
    assert "roc_auc" in first_fold["metrics"]


def test_summarize_cross_validation_mean_and_std():
    fold_results = [
        {
            "fold": 1,
            "metrics": {
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "roc_auc": 0.8,
            },
            "train_size": 80,
            "test_size": 20,
        },
        {
            "fold": 2,
            "metrics": {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "roc_auc": 1.0,
            },
            "train_size": 80,
            "test_size": 20,
        },
    ]

    report = summarize_cross_validation(fold_results)

    assert report["mean_metrics"]["accuracy"] == 0.9
    assert report["mean_metrics"]["f1"] == 0.9
    assert report["std_metrics"]["accuracy"] > 0


def test_flatten_cv_metrics():
    cv_report = {
        "mean_metrics": {
            "accuracy": 0.9,
            "f1": 0.85,
            "roc_auc": 0.92,
        },
        "std_metrics": {
            "accuracy": 0.02,
            "f1": 0.03,
            "roc_auc": 0.01,
        },
    }

    flattened = flatten_cv_metrics(cv_report)

    assert flattened["cv_accuracy_mean"] == 0.9
    assert flattened["cv_f1_mean"] == 0.85
    assert flattened["cv_roc_auc_mean"] == 0.92
    assert flattened["cv_accuracy_std"] == 0.02
    assert flattened["cv_f1_std"] == 0.03
    assert flattened["cv_roc_auc_std"] == 0.01