"""Input coercion so rejectkit accepts pandas, polars, or plain numpy."""

from __future__ import annotations

import numpy as np


def to_numpy_2d(X):
    """Coerce a 2D array-like (pandas/polars DataFrame, ndarray, list) to float ndarray."""
    if hasattr(X, "to_numpy"):  # pandas / polars DataFrame
        X = X.to_numpy()
    return np.asarray(X, dtype=float)


def to_numpy_1d(y):
    """Coerce a 1D array-like (pandas/polars Series, ndarray, list) to a 1D ndarray."""
    if hasattr(y, "to_numpy"):  # pandas / polars Series
        y = y.to_numpy()
    return np.asarray(y).ravel()
