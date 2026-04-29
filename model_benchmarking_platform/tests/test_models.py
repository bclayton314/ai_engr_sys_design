import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from models import build_model


def test_build_logistic_regression():
    model_config = {
        "name": "logreg",
        "type": "logistic_regression",
        "params": {
            "max_iter": 100,
            "solver": "liblinear",
        },
    }

    model = build_model(model_config)

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 100
    assert model.solver == "liblinear"


def test_build_decision_tree():
    model_config = {
        "name": "tree",
        "type": "decision_tree",
        "params": {
            "max_depth": 5,
            "random_state": 42,
        },
    }

    model = build_model(model_config)

    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 5
    assert model.random_state == 42


def test_build_random_forest():
    model_config = {
        "name": "forest",
        "type": "random_forest",
        "params": {
            "n_estimators": 10,
            "max_depth": 4,
            "random_state": 42,
        },
    }

    model = build_model(model_config)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    assert model.max_depth == 4
    assert model.random_state == 42


def test_build_model_unsupported_type():
    model_config = {
        "name": "bad_model",
        "type": "unsupported_model",
        "params": {},
    }

    with pytest.raises(ValueError, match="Unsupported model type"):
        build_model(model_config)