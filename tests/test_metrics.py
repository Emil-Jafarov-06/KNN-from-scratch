import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
    roc_auc_score,
)

from src.metrics import accuracy, precision, recall, f1_score, roc_auc


def test_metrics_match_sklearn():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.3, 0.85, 0.15])

    assert np.isclose(
        accuracy(y_true, y_pred),
        accuracy_score(y_true, y_pred),
    )

    assert np.isclose(
        precision(y_true, y_pred),
        precision_score(y_true, y_pred),
    )

    assert np.isclose(
        recall(y_true, y_pred),
        recall_score(y_true, y_pred),
    )

    assert np.isclose(
        f1_score(y_true, y_pred),
        sklearn_f1_score(y_true, y_pred),
    )

    assert np.isclose(
        roc_auc(y_true, y_score),
        roc_auc_score(y_true, y_score),
    )