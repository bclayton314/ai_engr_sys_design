import json
from pathlib import Path
import joblib

from utils import ensure_dir, utc_timestamp


class RunTracker:
    """
    Handles run directory creation and saving run outputs.
    """

    def __init__(self, experiment_name: str):
        self.run_id = f"{experiment_name}_{utc_timestamp()}"
        self.run_dir = ensure_dir(Path("runs") / self.run_id)
        self.artifact_dir = ensure_dir(Path("artifacts") / self.run_id)

    def save_config(self, config: dict) -> None:
        path = self.run_dir / "config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def save_metrics(self, metrics: dict) -> None:
        path = self.run_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def save_metadata(self, metadata: dict) -> None:
        path = self.run_dir / "metadata.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def save_model(self, model) -> str:
        model_path = self.artifact_dir / "model.joblib"
        joblib.dump(model, model_path)
        return str(model_path)