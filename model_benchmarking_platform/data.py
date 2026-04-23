from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_dataset(dataset_config: dict, random_seed: int):
    """
    Create a synthetic classification dataset and split it into train/test sets.
    """
    X, y = make_classification(
        n_samples=dataset_config["n_samples"],
        n_features=dataset_config["n_features"],
        n_informative=dataset_config["n_informative"],
        n_redundant=dataset_config["n_redundant"],
        n_classes=dataset_config["n_classes"],
        random_state=random_seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=dataset_config["test_size"],
        random_state=random_seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test