from data import load_dataset


def test_load_dataset_split_sizes():
    dataset_config = {
        "n_samples": 100,
        "n_features": 10,
        "n_informative": 5,
        "n_redundant": 2,
        "n_classes": 2,
        "test_size": 0.2,
    }

    X_train, X_test, y_train, y_test = load_dataset(
        dataset_config=dataset_config,
        random_seed=42,
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_load_dataset_feature_count():
    dataset_config = {
        "n_samples": 100,
        "n_features": 12,
        "n_informative": 6,
        "n_redundant": 2,
        "n_classes": 2,
        "test_size": 0.25,
    }

    X_train, X_test, _, _ = load_dataset(
        dataset_config=dataset_config,
        random_seed=42,
    )

    assert X_train.shape[1] == 12
    assert X_test.shape[1] == 12


def test_load_dataset_reproducible_with_same_seed():
    dataset_config = {
        "n_samples": 100,
        "n_features": 10,
        "n_informative": 5,
        "n_redundant": 2,
        "n_classes": 2,
        "test_size": 0.2,
    }

    first = load_dataset(dataset_config, random_seed=42)
    second = load_dataset(dataset_config, random_seed=42)

    X_train_1, X_test_1, y_train_1, y_test_1 = first
    X_train_2, X_test_2, y_train_2, y_test_2 = second

    assert (X_train_1 == X_train_2).all()
    assert (X_test_1 == X_test_2).all()
    assert (y_train_1 == y_train_2).all()
    assert (y_test_1 == y_test_2).all()