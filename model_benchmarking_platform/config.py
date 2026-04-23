import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {"experiment_name", "random_seed", "dataset", "model"}


def load_config(config_path: str) -> dict:
    """
    Load a JSON config file and perform minimal validation.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    missing = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    return config