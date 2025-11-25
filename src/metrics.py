import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_metrics(y_true, y_proba):
    """
    y_true: (N,) int labels
    y_proba: (N, C) predicted probabilities
    Returns dict with accuracy and macro AUC.
    """
    y_pred = y_proba.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)

    n_classes = y_proba.shape[1]
    try:
        y_onehot = np.eye(n_classes)[y_true]
        auc = roc_auc_score(y_onehot, y_proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "auc_macro": auc}
