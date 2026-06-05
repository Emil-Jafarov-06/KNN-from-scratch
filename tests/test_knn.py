import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.knn import KNN


def test_knn_matches_sklearn_classifier() -> None:
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(200, 5))
    y_train = rng.integers(0, 2, size=200)

    X_test = rng.normal(size=(40, 5))

    custom_model = KNN(
        k=5,
        metric="euclidean",
        task="classification",
        weights="uniform",
    )

    sklearn_model = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        weights="uniform",
    )

    custom_model.fit(X_train, y_train)
    sklearn_model.fit(X_train, y_train)

    custom_predictions = custom_model.predict(X_test)
    sklearn_predictions = sklearn_model.predict(X_test)

    custom_probabilities = custom_model.predict_proba(X_test)
    sklearn_probabilities = sklearn_model.predict_proba(X_test)

    assert np.array_equal(custom_predictions, sklearn_predictions)

    assert np.allclose(
        custom_probabilities,
        sklearn_probabilities,
        atol=1e-9,
    )


def test_predict_rejects_unfitted_model() -> None:
    model = KNN(k=1)

    with pytest.raises(ValueError):
        model.predict(np.array([[0.0]]))

def test_regression_prediction_with_euclidean_distance() -> None:
    X_train = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [10.0, 10.0],
    ])

    y_train = np.array([10.0, 20.0, 100.0])

    X_query = np.array([
        [0.0, 0.0],
    ])

    model = KNN(k=1, metric="euclidean", task="regression")
    model.fit(X_train, y_train)

    prediction = model.predict(X_query)

    assert np.allclose(prediction, np.array([10.0]))

def test_regression_prediction_uses_neighbor_mean() -> None:
    X_train = np.array([
        [0.0],
        [2.0],
        [10.0],
    ])

    y_train = np.array([10.0, 20.0, 100.0])

    X_query = np.array([
        [1.0],
    ])

    model = KNN(k=2, metric="euclidean", task="regression")
    model.fit(X_train, y_train)

    prediction = model.predict(X_query)

    assert np.allclose(prediction, np.array([15.0]))

def test_classification_prediction_uses_majority_vote() -> None:
    X_train = np.array([
        [0.0],
        [1.0],
        [2.0],
        [10.0],
    ])

    y_train = np.array([0, 1, 1, 0])

    X_query = np.array([
        [1.5],
    ])

    model = KNN(k=3, metric="euclidean", task="classification")
    model.fit(X_train, y_train)

    prediction = model.predict(X_query)

    assert np.array_equal(prediction, np.array([1]))

def test_predict_proba_returns_neighbor_proportions() -> None:
    X_train = np.array([
        [0.0],
        [1.0],
        [2.0],
        [10.0],
    ])

    y_train = np.array([0, 1, 1, 0])

    X_query = np.array([
        [1.5],
    ])

    model = KNN(k=3, metric="euclidean", task="classification")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_query)

    expected = np.array([
        [1 / 3, 2 / 3],
    ])

    assert np.allclose(probabilities, expected)
    assert np.allclose(probabilities.sum(axis=1), np.array([1.0]))

def test_minkowski_special_cases_match_other_metrics() -> None:
    X_train = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [7.0, 8.0],
    ])

    y_train = np.array([0, 1, 1])

    X_query = np.array([
        [1.0, 1.0],
        [5.0, 6.0],
    ])

    manhattan = KNN(k=1, metric="manhattan").fit(X_train, y_train)
    minkowski_q1 = KNN(k=1, metric="minkowski", q=1).fit(X_train, y_train)

    euclidean = KNN(k=1, metric="euclidean").fit(X_train, y_train)
    minkowski_q2 = KNN(k=1, metric="minkowski", q=2).fit(X_train, y_train)

    assert np.array_equal(
        manhattan.predict(X_query),
        minkowski_q1.predict(X_query),
    )

    assert np.array_equal(
        euclidean.predict(X_query),
        minkowski_q2.predict(X_query),
    )

def test_predict_proba_rejects_regression() -> None:
    X_train = np.array([
        [0.0],
        [1.0],
    ])

    y_train = np.array([10.0, 20.0])

    model = KNN(k=1, task="regression")
    model.fit(X_train, y_train)

    with pytest.raises(ValueError):
        model.predict_proba(np.array([[0.5]]))

def test_distance_weighting_gives_more_importance_to_closer_neighbors() -> None:
    X_train = np.array([[0.0], [4.0], [5.0]])
    y_train = np.array([0, 1, 1])
    X_query = np.array([[0.5]])

    uniform_model = KNN(k=3, weights="uniform")
    distance_model = KNN(k=3, weights="distance")

    uniform_model.fit(X_train, y_train)
    distance_model.fit(X_train, y_train)

    assert uniform_model.predict(X_query)[0] == 1
    assert distance_model.predict(X_query)[0] == 0


def test_distance_weighting_reduces_to_uniform_for_equidistant_neighbors() -> None:
    X_train = np.array([[-1.0], [1.0]])
    y_train = np.array([0, 1])
    X_query = np.array([[0.0]])

    uniform_model = KNN(k=2, weights="uniform")
    distance_model = KNN(k=2, weights="distance")

    uniform_model.fit(X_train, y_train)
    distance_model.fit(X_train, y_train)

    uniform_probabilities = uniform_model.predict_proba(X_query)
    distance_probabilities = distance_model.predict_proba(X_query)

    assert np.allclose(uniform_probabilities, distance_probabilities)
    assert np.allclose(distance_probabilities, np.array([[0.5, 0.5]]))


def test_distance_weighting_uses_weighted_average_for_regression() -> None:
    X_train = np.array([[0.0], [4.0]])
    y_train = np.array([10.0, 30.0])
    X_query = np.array([[1.0]])

    model = KNN(k=2, task="regression", weights="distance")
    model.fit(X_train, y_train)

    first_weight = 1 / (1.0 + 1e-8)
    second_weight = 1 / (3.0 + 1e-8)
    expected_prediction = (10.0 * first_weight + 30.0 * second_weight) / (first_weight + second_weight)

    assert np.allclose(model.predict(X_query), np.array([expected_prediction]))