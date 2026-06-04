import numpy as np
from sklearn.model_selection import train_test_split

from src.splits import stratified_kfold, stratified_split


def test_stratified_split_shapes_and_total_size():
    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 60 + [1] * 40)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    assert X_train.shape[0] == 60
    assert X_val.shape[0] == 20
    assert X_test.shape[0] == 20

    assert len(y_train) + len(y_val) + len(y_test) == len(y)


def test_stratified_split_is_reproducible():
    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 60 + [1] * 40)

    first_split = stratified_split(X, y, seed=42)
    second_split = stratified_split(X, y, seed=42)

    for first_array, second_array in zip(first_split, second_split):
        assert np.array_equal(first_array, second_array)


def test_stratified_split_preserves_class_proportions():
    X = np.arange(2000).reshape(1000, 2)
    y = np.array([0] * 620 + [1] * 380)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    overall_ratio = np.mean(y == 1)

    assert abs(np.mean(y_train == 1) - overall_ratio) <= 0.005
    assert abs(np.mean(y_val == 1) - overall_ratio) <= 0.005
    assert abs(np.mean(y_test == 1) - overall_ratio) <= 0.005


def test_stratified_split_matches_sklearn_proportions():
    X = np.arange(2000).reshape(1000, 2)
    y = np.array([0] * 620 + [1] * 380)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, seed=42)

    X_train_sk, X_temp_sk, y_train_sk, y_temp_sk = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )

    X_val_sk, X_test_sk, y_val_sk, y_test_sk = train_test_split(
        X_temp_sk,
        y_temp_sk,
        test_size=0.5,
        random_state=42,
        stratify=y_temp_sk,
    )

    assert np.isclose(
        np.mean(y_train == 1),
        np.mean(y_train_sk == 1),
    )

    assert np.isclose(
        np.mean(y_val == 1),
        np.mean(y_val_sk == 1),
    )

    assert np.isclose(
        np.mean(y_test == 1),
        np.mean(y_test_sk == 1),
    )

def test_stratified_kfold_returns_correct_number_of_folds() -> None:
    X = np.arange(40).reshape(20, 2)
    y = np.array([0] * 12 + [1] * 8)

    folds = list(stratified_kfold(X, y, K=4, seed=42))

    assert len(folds) == 4

def test_stratified_kfold_validation_sets_cover_all_samples_once() -> None:
    X = np.arange(40).reshape(20, 2)
    y = np.array([0] * 12 + [1] * 8)

    folds = list(stratified_kfold(X, y, K=4, seed=42))

    validation_rows = []

    for _, X_val, _, _ in folds:
        validation_rows.extend(X_val.tolist())

    validation_rows = np.array(validation_rows)

    assert validation_rows.shape[0] == X.shape[0]
    assert np.unique(validation_rows, axis=0).shape[0] == X.shape[0]

def test_stratified_kfold_train_and_validation_do_not_overlap() -> None:
    X = np.arange(40).reshape(20, 2)
    y = np.array([0] * 12 + [1] * 8)

    folds = list(stratified_kfold(X, y, K=4, seed=42))

    for X_train, X_val, _, _ in folds:
        train_rows = {tuple(row) for row in X_train}
        val_rows = {tuple(row) for row in X_val}

        assert train_rows.isdisjoint(val_rows)

def test_stratified_kfold_preserves_class_proportions() -> None:
    X = np.arange(200).reshape(100, 2)
    y = np.array([0] * 60 + [1] * 40)

    folds = list(stratified_kfold(X, y, K=5, seed=42))

    full_positive_proportion = np.mean(y == 1)

    for _, _, _, y_val in folds:
        val_positive_proportion = np.mean(y_val == 1)

        assert np.isclose(
            val_positive_proportion,
            full_positive_proportion,
            atol=0.01,
        )