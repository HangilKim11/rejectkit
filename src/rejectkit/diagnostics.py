"""Lightweight evaluation and drift metrics used across rejectkit.

Dependency-free (numpy + scikit-learn + pandas) and using the ``1 = bad``
convention, so ``y_score`` is always interpreted as P(bad).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._compat import to_numpy_1d, to_numpy_2d


def _check_arrays(y_true, y_score):
    y_true = to_numpy_1d(y_true)
    y_score = to_numpy_1d(y_score).astype(float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    return y_true, y_score


def auc(y_true, y_score) -> float:
    """Area under the ROC curve."""
    from sklearn.metrics import roc_auc_score

    y_true, y_score = _check_arrays(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    """Gini coefficient, ``2 * AUC - 1``."""
    return 2.0 * auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic: max separation of good/bad CDFs."""
    from sklearn.metrics import roc_curve

    y_true, y_score = _check_arrays(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two distributions.

    Bins are quantiles of ``expected``. Rule of thumb: < 0.1 no shift,
    0.1-0.25 moderate, > 0.25 major.
    """
    expected = to_numpy_1d(expected).astype(float)
    actual = to_numpy_1d(actual).astype(float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), eps, None)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def feature_drift(X_accept, X_reject, n_bins: int = 10) -> pd.Series:
    """Per-feature PSI between the accept and reject populations.

    A quick read on how unrepresentative your accepts are: large values flag
    features whose distribution differs most between accepts and rejects.
    """
    cols = list(X_accept.columns) if hasattr(X_accept, "columns") else None
    A = to_numpy_2d(X_accept)
    R = to_numpy_2d(X_reject)
    if cols is None:
        cols = [f"x{i + 1}" for i in range(A.shape[1])]
    values = {c: psi(A[:, i], R[:, i], n_bins=n_bins) for i, c in enumerate(cols)}
    return pd.Series(values, name="psi").sort_values(ascending=False)


def swap_set(
    y_true,
    score_reference,
    score_challenger,
    cutoff_reference,
    cutoff_challenger,
    lower_is_safer: bool = True,
) -> pd.DataFrame:
    """Swap-set analysis between a reference and a challenger scorecard.

    Compares which applicants each policy accepts (accept if the score is on the
    safe side of its cutoff) and reports counts and bad rates for the four
    groups: kept-accept, kept-reject, swap-in (reference rejects, challenger
    accepts) and swap-out (reference accepts, challenger rejects).
    """
    y = to_numpy_1d(y_true).astype(int)
    sr = to_numpy_1d(score_reference).astype(float)
    sc = to_numpy_1d(score_challenger).astype(float)
    acc_r = sr <= cutoff_reference if lower_is_safer else sr >= cutoff_reference
    acc_c = sc <= cutoff_challenger if lower_is_safer else sc >= cutoff_challenger
    groups = {
        "kept_accept": acc_r & acc_c,
        "swap_out": acc_r & ~acc_c,
        "swap_in": ~acc_r & acc_c,
        "kept_reject": ~acc_r & ~acc_c,
    }
    rows = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        rows[name] = {"n": n, "bad_rate": float(y[mask].mean()) if n else float("nan")}
    return pd.DataFrame(rows).T[["n", "bad_rate"]]
