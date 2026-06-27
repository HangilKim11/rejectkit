"""Synthetic data generators for examples, tests and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _solve_intercept(logit: np.ndarray, target: float, iters: int = 60) -> float:
    """Bisection for the intercept that yields a target mean event rate."""
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _sigmoid(logit + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_credit_data(
    n_samples: int = 3000,
    n_features: int = 8,
    bad_rate: float = 0.3,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic, fully labelled credit dataset.

    Returns
    -------
    X : pandas.DataFrame, shape (n_samples, n_features)
    y : pandas.Series
        ``1 = bad`` (default), ``0 = good``. The mean of ``y`` is approximately
        ``bad_rate``.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    beta = rng.normal(size=n_features)
    beta[n_features // 2:] *= 0.2  # second half weakly informative
    logit = X @ beta
    logit = 1.5 * (logit - logit.mean()) / (logit.std() + 1e-12)
    intercept = _solve_intercept(logit, bad_rate)
    p_bad = _sigmoid(logit + intercept)
    y = (rng.random(n_samples) < p_bad).astype(int)
    cols = [f"x{i + 1}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="bad")


def make_accept_reject(
    n_samples: int = 3000,
    n_features: int = 8,
    bad_rate: float = 0.3,
    accept_rate: float = 0.6,
    selection: str = "mnar",
    selection_strength: float = 1.5,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate accept/reject data with the reject labels hidden.

    Returns
    -------
    X_accept, y_accept : the accepted applicants and their observed outcomes.
    X_reject : the rejected applicants (features only).
    y_reject_true : the rejects' true outcomes — **hidden** in practice, returned
        only so demos and tests can quantify how well a method recovers them.
    """
    X, y = make_credit_data(n_samples, n_features, bad_rate, random_state)
    rng = np.random.default_rng(random_state + 1)
    Xv = X.to_numpy()
    Xs = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-12)
    w = rng.normal(size=n_features)
    feat = Xs @ w
    feat = feat / (feat.std() + 1e-12)
    noise = rng.normal(size=n_samples)
    if selection == "mnar":
        z = feat + selection_strength * (0.5 - y.to_numpy()) * 2.0 + noise
    elif selection == "mar":
        z = selection_strength * feat + noise
    else:
        raise ValueError("selection must be 'mar' or 'mnar'.")
    accepted = z >= np.quantile(z, 1.0 - accept_rate)
    return (
        X[accepted].reset_index(drop=True),
        y[accepted].reset_index(drop=True),
        X[~accepted].reset_index(drop=True),
        y[~accepted].reset_index(drop=True),
    )
