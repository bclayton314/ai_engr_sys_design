from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_dataset(dataset_config: dict, random_seed: int):
    """
    Create a synthetic classification dataset.

    This returns the full dataset before any splitting so it can be used for
    both holdout evaluation and cross-validation.
    """
    X, y = make_classification(
        n_samples=dataset_config["n_samples"],
        n_features=dataset_config["n_features"],
        n_informative=dataset_config["n_informative"],
        n_redundant=dataset_config["n_redundant"],
        n_classes=dataset_config["n_classes"],
        random_state=random_seed,
    )

    return X, y


def split_dataset(X, y, dataset_config: dict, random_seed: int):
    """
    Split a dataset into train/test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=dataset_config["test_size"],
        random_state=random_seed,
        stratify=y,
    )


def load_dataset(dataset_config: dict, random_seed: int):
    """
    Backward-compatible helper that creates and splits a dataset.
    """
    X, y = create_dataset(
        dataset_config=dataset_config,
        random_seed=random_seed,
    )

    return split_dataset(
        X=X,
        y=y,
        dataset_config=dataset_config,
        random_seed=random_seed,
    )