from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Compute standard binary classification metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def compute_evaluation_report(y_true, y_pred, y_prob) -> dict:
    """
    Compute standard binary classification metrics and return evaluation report.
    """
    summary_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    return {
        "summary_metrics": summary_metrics,
    }