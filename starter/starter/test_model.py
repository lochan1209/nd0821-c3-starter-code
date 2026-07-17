import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import train_model, inference, compute_model_metrics


def test_train_model():
    X = np.array([
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 0, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_inference():
    X_train = np.array([
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1]
    ])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    X_test = np.array([
        [0, 1],
        [1, 1]
    ])

    preds = inference(model, X_test)

    assert len(preds) == len(X_test)


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert type(precision) == float
    assert type(recall) == float
    assert type(fbeta) == float
