"""Iterative reclassification reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ArrayTriple, BaseRejectInferencer


class Reclassification(BaseRejectInferencer):
    """Iterative reclassification.

    Fits a good/bad model on the accepts, hard-labels the rejects by
    thresholding P(bad), adds them to the training data, refits, and repeats.
    Labels may change between iterations until they stabilise or ``n_iter`` is
    reached.

    Parameters
    ----------
    threshold : float, default=0.5
    n_iter : int, default=3
    """

    def __init__(self, base_estimator: BaseEstimator | None = None,
                 threshold: float = 0.5, n_iter: int = 3):
        super().__init__(base_estimator=base_estimator)
        self.threshold = threshold
        self.n_iter = n_iter

    def resample(self) -> ArrayTriple:
        Xa, ya, Xr = self.X_accept_, self.y_accept_, self.X_reject_
        model = self.scorer_
        y_reject = (model.predict_proba(Xr)[:, 1] >= self.threshold).astype(int)
        for _ in range(max(self.n_iter - 1, 0)):
            model = self._make_base()
            model.fit(np.vstack([Xa, Xr]), np.concatenate([ya, y_reject]))
            new = (model.predict_proba(Xr)[:, 1] >= self.threshold).astype(int)
            stable = np.array_equal(new, y_reject)
            y_reject = new
            if stable:
                break
        X = np.vstack([Xa, Xr])
        y = np.concatenate([ya, y_reject])
        return X, y, np.ones(X.shape[0])
