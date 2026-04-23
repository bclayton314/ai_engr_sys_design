from sklearn.linear_model import LogisticRegression


def build_model(model_config: dict):
    """
    Create and return a model instance based on config.
    """
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type == "logistic_regression":
        return LogisticRegression(**params)

    raise ValueError(f"Unsupported model type: {model_type}")