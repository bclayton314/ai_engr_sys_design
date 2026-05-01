import json
import os

from sklearn.linear_model import LogisticRegression

from tracker import RunTracker


def test_run_tracker_creates_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    tracker = RunTracker(
        experiment_name="test_experiment",
        model_name="test_model",
    )

    assert tracker.run_dir.exists()
    assert tracker.artifact_dir.exists()
    assert "test_experiment" in tracker.run_id
    assert "test_model" in tracker.run_id


def test_run_tracker_saves_config_metrics_reports_and_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    tracker = RunTracker(
        experiment_name="test_experiment",
        model_name="test_model",
    )

    config = {"hello": "config"}
    metrics = {"accuracy": 0.9}
    evaluation_report = {
        "summary_metrics": metrics,
        "confusion_matrix": [[9, 1], [2, 8]],
        "classification_report": {},
    }
    cv_report = {
        "folds": [],
        "mean_metrics": {"accuracy": 0.9},
        "std_metrics": {"accuracy": 0.02},
    }
    metadata = {"model_name": "test_model"}

    tracker.save_config(config)
    tracker.save_metrics(metrics)
    tracker.save_evaluation_report(evaluation_report)
    tracker.save_cross_validation_report(cv_report)
    tracker.save_metadata(metadata)

    config_path = tracker.run_dir / "config.json"
    metrics_path = tracker.run_dir / "metrics.json"
    evaluation_report_path = tracker.run_dir / "evaluation_report.json"
    cv_report_path = tracker.run_dir / "cross_validation_report.json"
    metadata_path = tracker.run_dir / "metadata.json"

    assert config_path.exists()
    assert metrics_path.exists()
    assert evaluation_report_path.exists()
    assert cv_report_path.exists()
    assert metadata_path.exists()

    with config_path.open("r", encoding="utf-8") as f:
        loaded_config = json.load(f)

    with metrics_path.open("r", encoding="utf-8") as f:
        loaded_metrics = json.load(f)

    with evaluation_report_path.open("r", encoding="utf-8") as f:
        loaded_evaluation_report = json.load(f)

    with cv_report_path.open("r", encoding="utf-8") as f:
        loaded_cv_report = json.load(f)

    with metadata_path.open("r", encoding="utf-8") as f:
        loaded_metadata = json.load(f)

    assert loaded_config == config
    assert loaded_metrics == metrics
    assert loaded_evaluation_report == evaluation_report
    assert loaded_cv_report == cv_report
    assert loaded_metadata == metadata


def test_run_tracker_saves_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    tracker = RunTracker(
        experiment_name="test_experiment",
        model_name="test_model",
    )

    model = LogisticRegression()
    model_path = tracker.save_model(model)

    assert os.path.exists(model_path)
    assert model_path.endswith("model.joblib")