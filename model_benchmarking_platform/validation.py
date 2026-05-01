from statistics import mean, stdev

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from metrics import compute_classification_metrics


def run_cross_validation(
    model,
    X,
    y,
    n_splits: int,
    shuffle: bool,
    random_seed: int,
) -> dict:
    """
    Run stratified k-fold cross-validation and return fold-level metrics,
    mean metrics, and standard deviation metrics.
    """
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_seed if shuffle else None,
    )

    fold_results = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        fold_model = clone(model)

        X_train_fold = X[train_index]
        X_test_fold = X[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]

        fold_model.fit(X_train_fold, y_train_fold)

        y_pred = fold_model.predict(X_test_fold)

        if not hasattr(fold_model, "predict_proba"):
            raise ValueError(
                "Cross-validation requires models to support predict_proba "
                "for ROC-AUC calculation."
            )

        y_prob = fold_model.predict_proba(X_test_fold)[:, 1]

        fold_metrics = compute_classification_metrics(
            y_true=y_test_fold,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        fold_results.append(
            {
                "fold": fold_index,
                "metrics": fold_metrics,
                "train_size": int(len(train_index)),
                "test_size": int(len(test_index)),
            }
        )

    return summarize_cross_validation(fold_results)


def summarize_cross_validation(fold_results: list[dict]) -> dict:
    """
    Summarize cross-validation metrics across folds.
    """
    metric_names = fold_results[0]["metrics"].keys()

    mean_metrics = {}
    std_metrics = {}

    for metric_name in metric_names:
        values = [fold["metrics"][metric_name] for fold in fold_results]

        mean_metrics[metric_name] = float(mean(values))

        if len(values) > 1:
            std_metrics[metric_name] = float(stdev(values))
        else:
            std_metrics[metric_name] = 0.0

    return {
        "folds": fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def flatten_cv_metrics(cv_report: dict) -> dict:
    """
    Flatten cross-validation metrics for leaderboard usage.

    Example:
    {
      "cv_f1_mean": 0.88,
      "cv_f1_std": 0.03
    }
    """
    flattened = {}

    for metric_name, value in cv_report["mean_metrics"].items():
        flattened[f"cv_{metric_name}_mean"] = value

    for metric_name, value in cv_report["std_metrics"].items():
        flattened[f"cv_{metric_name}_std"] = value

    return flattened