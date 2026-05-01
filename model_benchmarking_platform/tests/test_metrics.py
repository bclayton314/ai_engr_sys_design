from metrics import compute_classification_metrics, compute_evaluation_report


def test_compute_classification_metrics_perfect_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_classification_metrics_bad_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 0, 0, 1]
    y_prob = [0.9, 0.1, 0.2, 0.8]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_compute_evaluation_report_perfect_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2]

    report = compute_evaluation_report(y_true, y_pred, y_prob)
    metrics = report["summary_metrics"]

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_evaluation_report_bad_predictions():
    y_true = [0, 1, 1, 0]
    y_pred = [1, 0, 0, 1]
    y_prob = [0.9, 0.1, 0.2, 0.8]

    report = compute_evaluation_report(y_true, y_pred, y_prob)
    metrics = report["summary_metrics"]

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_compute_classification_metrics_contains_expected_keys():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_prob = [0.2, 0.8, 0.4, 0.3]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }