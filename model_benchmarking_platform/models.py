from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_model(model_config: dict):
    """
    Create and return a model instance based on a model config.
    """
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type == "logistic_regression":
        return LogisticRegression(**params)

    if model_type == "decision_tree":
        return DecisionTreeClassifier(**params)

    if model_type == "random_forest":
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model type: {model_type}")