import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {
    "experiment_name",
    "random_seed",
    "dataset",
    "models",
}


def load_config(config_path: str) -> dict:
    """
    Load a JSON config file and validate the basic experiment structure.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """
    Validate the top-level config fields and model list.
    """
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    if not isinstance(config["models"], list):
        raise ValueError("'models' must be a list.")

    if len(config["models"]) == 0:
        raise ValueError("'models' must contain at least one model config.")

    for model_config in config["models"]:
        required_model_keys = {"name", "type", "params"}
        missing_model_keys = required_model_keys - set(model_config.keys())

        if missing_model_keys:
            raise ValueError(
                f"Model config is missing keys: {sorted(missing_model_keys)}"
            )