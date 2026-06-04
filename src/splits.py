import numpy as np
from collections.abc import Iterator

def stratified_split (X : np.ndarray, y : np.ndarray, train_frac =0.6, val_frac =0.2,
                      test_frac =0.2 , seed =42) -> tuple[np.ndarray, ...]:
    """Split X and y into stratified train, validation, and test subsets."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be of type np.ndarray!")
    if X.ndim != 2:
        raise TypeError("X must have 2 dimensions!")
    if y.ndim != 1:
        raise TypeError("y must have 1 dimension!")
    if X.shape[0] == 0:
        raise ValueError("X and y must not be empty!")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows!")
    if train_frac < 0 or val_frac < 0 or test_frac < 0:
        raise ValueError("train_frac, val_frac, and test_frac must be non-negative!")
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1!")

    train_indexes, val_indexes, test_indexes = list(), list(), list()

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    for class_value in classes:
        class_idx = np.where(y == class_value)[0]
        rng.shuffle(class_idx)

        train_size = int(np.round(len(class_idx) * train_frac))
        val_size = int(np.round(len(class_idx) * val_frac))
        test_size = int(len(class_idx) - train_size - val_size)

        assert train_size + val_size + test_size == len(class_idx)

        train_indexes.append(class_idx[:train_size])
        val_indexes.append(class_idx[train_size: train_size+val_size])
        test_indexes.append(class_idx[train_size+val_size:])

    train_indexes = np.concatenate(train_indexes)
    val_indexes = np.concatenate(val_indexes)
    test_indexes = np.concatenate(test_indexes)

    rng.shuffle(train_indexes)
    rng.shuffle(val_indexes)
    rng.shuffle(test_indexes)

    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_val = X[val_indexes]
    y_val = y[val_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_val, X_test, y_train, y_val, y_test

Fold = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def stratified_kfold(
    X: np.ndarray,
    y: np.ndarray,
    K: int = 5,
    seed: int = 42,
) -> Iterator[Fold]:
    """Yield K stratified training and validation folds."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be NumPy arrays.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] == 0:
        raise ValueError("X and y must not be empty.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if type(K) is not int or K < 2:
        raise ValueError("K must be an integer greater than or equal to 2.")

    _, class_counts = np.unique(y, return_counts=True)
    if np.any(class_counts < K):
        raise ValueError("Each class must contain at least K samples.")

    rng = np.random.default_rng(seed)

    validation_folds = [[] for _ in range(K)]

    for label in np.unique(y):
        class_idx = np.where(y == label)[0]
        rng.shuffle(class_idx)

        class_parts = np.array_split(class_idx, K)
        for i in range(K):
            validation_folds[i].extend(class_parts[i])

    all_indices = np.arange(X.shape[0])

    for fold_number in range(K):
        val_idx = np.array(validation_folds[fold_number])
        rng.shuffle(val_idx)

        train_mask = np.ones(X.shape[0], dtype=bool)
        train_mask[val_idx] = False

        train_idx = all_indices[train_mask]
        rng.shuffle(train_idx)

        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]