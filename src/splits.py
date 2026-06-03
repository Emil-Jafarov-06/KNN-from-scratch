import numpy as np

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

