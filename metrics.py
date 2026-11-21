"""Group-disparity metrics and the validation-F1 threshold search."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def fairness_gaps(pred, labels, group):
    """Statistical-parity and equal-opportunity gaps between groups 0 and 1.

    SP = |P(y_hat=1 | s=0) - P(y_hat=1 | s=1)|
    EO = |P(y_hat=1 | y=1, s=0) - P(y_hat=1 | y=1, s=1)|
    """
    g0, g1 = group == 0, group == 1
    sp = abs(pred[g0].mean() - pred[g1].mean())
    tpr0 = ((pred[g0] == 1) & (labels[g0] == 1)).sum() / max(labels[g0].sum(), 1)
    tpr1 = ((pred[g1] == 1) & (labels[g1] == 1)).sum() / max(labels[g1].sum(), 1)
    return float(sp), float(abs(tpr0 - tpr1))


def safe_auc(labels, scores):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:  # only one class present
        return 0.5


def best_f1_threshold(labels, scores, grid):
    """Threshold on `grid` (start, stop, step) with the highest F1; first one wins ties."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(*grid):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t
