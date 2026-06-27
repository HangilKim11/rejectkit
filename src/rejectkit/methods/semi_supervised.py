"""Semi-supervised (self-training) reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ArrayTriple, BaseRejectInferencer


class SelfLearning(BaseRejectInferencer):
    """Self-training reject inference.

    Treats rejects as unlabelled. Starting from a model fit on the accepts, it
    iteratively pseudo-labels the rejects it is most confident about
    (``P(bad) >= threshold`` -> bad, ``P(bad) <= 1 - threshold`` -> good),
    refits on the accepts plus newly labelled rejects, and repeats. Rejects
    that never cross the confidence band are excluded from the final sample.

    Parameters
    ----------
    threshold : float, default=0.75
    max_iter : int, default=10
    """

    def __init__(self, base_estimator: BaseEstimator | None = None,
                 threshold: float = 0.75, max_iter: int = 10):
        super().__init__(base_estimator=base_estimator)
        self.threshold = threshold
        self.max_iter = max_iter

    def resample(self) -> ArrayTriple:
        Xa, ya, Xr = self.X_accept_, self.y_accept_, self.X_reject_
        n_r = Xr.shape[0]
        labels = np.full(n_r, -1, dtype=int)
        lo = 1.0 - self.threshold
        model = self.scorer_
        for _ in range(self.max_iter):
            unl = labels == -1
            if not unl.any():
                break
            p = model.predict_proba(Xr[unl])[:, 1]
            local = np.where(unl)[0]
            newly = False
            bad = local[p >= self.threshold]
            good = local[p <= lo]
            if bad.size:
                labels[bad] = 1
                newly = True
            if good.size:
                labels[good] = 0
                newly = True
            if not newly:
                break
            keep = labels != -1
            model = self._make_base()
            model.fit(np.vstack([Xa, Xr[keep]]), np.concatenate([ya, labels[keep]]))
        keep = labels != -1
        if keep.any():
            X = np.vstack([Xa, Xr[keep]])
            y = np.concatenate([ya, labels[keep]])
        else:
            X, y = Xa.copy(), ya.copy()
        return X, y, np.ones(X.shape[0])
