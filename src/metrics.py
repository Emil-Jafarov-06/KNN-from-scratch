import numpy as np


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate two one-dimensional arrays with matching lengths."""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays!")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be one-dimensional!")
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have the same length!")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty!")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the proportion of correct predictions."""
    _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label=1,
) -> float:
    """Return TP / (TP + FP) for the selected positive label."""
    _validate_inputs(y_true, y_pred)
    true_positive = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    false_positive = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))

    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0
    return float(true_positive / denominator)

def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label=1,
) -> float:
    """Return TP / (TP + FN) for the selected positive label."""
    _validate_inputs(y_true, y_pred)

    true_positive = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    false_negative = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))

    denominator = true_positive + false_negative
    if denominator == 0:
        return 0.0
    return float(true_positive / denominator)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label=1,
) -> float:
    """Return the harmonic mean of precision and recall."""
    prec = precision(y_true, y_pred, positive_label)
    rec = recall(y_true, y_pred, positive_label)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return ROC-AUC for binary classification scores."""
    _validate_inputs(y_true, y_score)
    classes = np.unique(y_true)

    if len(classes) != 2:
        raise ValueError("y_true must contain exactly two classes!")

    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("y_true must contain binary labels 0 and 1!")

    positive_count = np.sum(y_true == 1)
    negative_count = np.sum(y_true == 0)

    thresholds = np.unique(y_score)
    thresholds = thresholds[::-1]
    thresholds = np.concatenate(([np.inf], thresholds))

    false_positive_rates = []
    true_positive_rates = []

    for threshold in thresholds:
        predicted_positive = y_score >= threshold

        true_positive = np.sum(predicted_positive & (y_true == 1))
        false_positive = np.sum(predicted_positive & (y_true == 0))
        true_positive_rate = true_positive / positive_count
        false_positive_rate = false_positive / negative_count

        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    false_positive_rates = np.array(false_positive_rates)
    true_positive_rates = np.array(true_positive_rates)

    return float(np.trapezoid(true_positive_rates, false_positive_rates))