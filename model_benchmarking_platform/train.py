from datetime import datetime, UTC

from data import load_dataset
from metrics import compute_classification_metrics
from models import build_model
from tracker import RunTracker


def run_training(config: dict) -> dict:
    """
    Execute one full training + evaluation run.
    """
    experiment_name = config["experiment_name"]
    random_seed = config["random_seed"]

    tracker = RunTracker(experiment_name=experiment_name)
    tracker.save_config(config)

    X_train, X_test, y_train, y_test = load_dataset(
        dataset_config=config["dataset"],
        random_seed=random_seed,
    )

    model = build_model(config["model"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    tracker.save_metrics(metrics)

    model_path = tracker.save_model(model)

    metadata = {
        "run_id": tracker.run_id,
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_type": config["model"]["type"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "model_artifact_path": model_path,
    }
    tracker.save_metadata(metadata)

    return {
        "run_id": tracker.run_id,
        "metrics": metrics,
        "model_path": model_path,
        "run_dir": str(tracker.run_dir),
    }