"""Base class shared by all reject inference methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array

from ._compat import to_numpy_1d, to_numpy_2d

ArrayTriple = tuple[np.ndarray, np.ndarray, np.ndarray]


class BaseRejectInferencer(BaseEstimator, ABC):
    """Abstract base class for reject inference methods.

    A reject inferencer takes the accepted applicants ``(X_accept, y_accept)``,
    whose good/bad outcome is known, together with the rejected applicants
    ``X_reject``, whose outcome is unknown, and produces an augmented, weighted
    training sample ``(X, y, sample_weight)`` that approximates the full
    "through-the-door" population.

    The label convention is ``1 = bad`` (the event of interest, e.g. default)
    and ``0 = good``, matching ``predict_proba(...)[:, 1] = P(bad)``.

    Accepts pandas, polars, or numpy inputs.

    Parameters
    ----------
    base_estimator : sklearn classifier, optional
        Probabilistic model used internally by the method. Must implement
        ``predict_proba``. Defaults to LogisticRegression.
    """

    def __init__(self, base_estimator: BaseEstimator | None = None):
        self.base_estimator = base_estimator

    def _make_base(self) -> BaseEstimator:
        if self.base_estimator is None:
            return LogisticRegression(max_iter=1000)
        return clone(self.base_estimator)

    def _validate(self, X_accept, y_accept, X_reject):
        X_accept = check_array(to_numpy_2d(X_accept), dtype=float)
        X_reject = check_array(to_numpy_2d(X_reject), dtype=float)
        y_accept = to_numpy_1d(y_accept)
        try:
            y_accept = y_accept.astype(int)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            raise ValueError("y_accept must be coercible to integers 0/1.") from exc
        if X_accept.shape[1] != X_reject.shape[1]:
            raise ValueError(
                f"X_accept has {X_accept.shape[1]} features but X_reject has "
                f"{X_reject.shape[1]}."
            )
        classes = set(np.unique(y_accept).tolist())
        if classes != {0, 1}:
            raise ValueError(
                "y_accept must be binary with both labels present, where "
                f"1 = bad (event) and 0 = good. Got classes {sorted(classes)}."
            )
        return X_accept, y_accept, X_reject

    def _reject_bad_proba(self) -> np.ndarray:
        """P(bad) for rejected applicants from the good/bad scorer."""
        return self.scorer_.predict_proba(self.X_reject_)[:, 1]

    def fit(self, X_accept, y_accept, X_reject):
        """Fit any internal models on the accepts (and rejects, if needed)."""
        X_accept, y_accept, X_reject = self._validate(X_accept, y_accept, X_reject)
        self.X_accept_ = X_accept
        self.y_accept_ = y_accept
        self.X_reject_ = X_reject
        self.n_features_in_ = X_accept.shape[1]
        self._fit()
        self.is_fitted_ = True
        return self

    def _fit(self) -> None:
        """Hook for method-specific fitting; default fits the good/bad scorer."""
        self.scorer_ = self._make_base()
        self.scorer_.fit(self.X_accept_, self.y_accept_)

    @abstractmethod
    def resample(self) -> ArrayTriple:
        """Return the augmented training sample ``(X, y, sample_weight)``."""
        raise NotImplementedError

    def fit_resample(self, X_accept, y_accept, X_reject) -> ArrayTriple:
        """Convenience: :meth:`fit` followed by :meth:`resample`."""
        return self.fit(X_accept, y_accept, X_reject).resample()
