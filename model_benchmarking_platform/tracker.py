import json
from pathlib import Path
import joblib

from utils import ensure_dir, utc_timestamp


class RunTracker:
    """
    Handles run directory creation and saving run outputs.
    """

    def __init__(self, experiment_name: str, model_name: str):
        self.run_id = f"{experiment_name}_{model_name}_{utc_timestamp()}"
        self.run_dir = ensure_dir(Path("runs") / self.run_id)
        self.artifact_dir = ensure_dir(Path("artifacts") / self.run_id)

    def save_config(self, config: dict) -> None:
        path = self.run_dir / "config.json"
        self._save_json(path, config)

    def save_metrics(self, metrics: dict) -> None:
        path = self.run_dir / "metrics.json"
        self._save_json(path, metrics)

    def save_evaluation_report(self, report: dict) -> None:
        path = self.run_dir / "evaluation_report.json"
        self._save_json(path, report)

    def save_cross_validation_report(self, report: dict) -> None:
        path = self.run_dir / "cross_validation_report.json"
        self._save_json(path, report)

    def save_metadata(self, metadata: dict) -> None:
        path = self.run_dir / "metadata.json"
        self._save_json(path, metadata)

    def save_model(self, model) -> str:
        model_path = self.artifact_dir / "model.joblib"
        joblib.dump(model, model_path)
        return str(model_path)

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)