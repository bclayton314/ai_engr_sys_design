import json
import pytest

from config import load_config, validate_config


def test_load_config_success(tmp_path):
    config_data = {
        "experiment_name": "test_experiment",
        "random_seed": 42,
        "dataset": {
            "n_samples": 100,
            "n_features": 10,
            "n_informative": 5,
            "n_redundant": 2,
            "n_classes": 2,
            "test_size": 0.2,
        },
        "models": [
            {
                "name": "logistic_regression_test",
                "type": "logistic_regression",
                "params": {
                    "max_iter": 100,
                    "solver": "liblinear",
                },
            }
        ],
    }

    config_path = tmp_path / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f)

    loaded = load_config(str(config_path))

    assert loaded["experiment_name"] == "test_experiment"
    assert loaded["random_seed"] == 42
    assert len(loaded["models"]) == 1


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("missing_config.json")


def test_validate_config_missing_required_key():
    invalid_config = {
        "experiment_name": "bad_config",
        "random_seed": 42,
        "dataset": {},
    }

    with pytest.raises(ValueError, match="Missing required config keys"):
        validate_config(invalid_config)


def test_validate_config_models_must_be_list():
    invalid_config = {
        "experiment_name": "bad_config",
        "random_seed": 42,
        "dataset": {},
        "models": {
            "name": "not_a_list"
        },
    }

    with pytest.raises(ValueError, match="'models' must be a list"):
        validate_config(invalid_config)


def test_validate_config_models_cannot_be_empty():
    invalid_config = {
        "experiment_name": "bad_config",
        "random_seed": 42,
        "dataset": {},
        "models": [],
    }

    with pytest.raises(ValueError, match="at least one model"):
        validate_config(invalid_config)


def test_validate_config_model_missing_name():
    invalid_config = {
        "experiment_name": "bad_config",
        "random_seed": 42,
        "dataset": {},
        "models": [
            {
                "type": "logistic_regression",
                "params": {},
            }
        ],
    }

    with pytest.raises(ValueError, match="Model config is missing keys"):
        validate_config(invalid_config)