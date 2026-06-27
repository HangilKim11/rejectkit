"""Inverse-propensity reweighting (IPW) reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ArrayTriple, BaseRejectInferencer


class Reweighting(BaseRejectInferencer):
    """Inverse-probability-of-acceptance reweighting.

    Rather than inventing labels for rejects, this fits a *selection model*
    that separates accepts from rejects, then trains only on the accepts, each
    weighted by ``1 / P(accept | x)``. Accepts that look like rejects are
    up-weighted, correcting the sample-selection bias under a
    missing-at-random assumption. No reject labels are fabricated.

    Parameters
    ----------
    clip : float, default=0.01
        Acceptance probabilities are clipped to ``[clip, 1]`` before inversion
        to bound the weights.
    """

    def __init__(self, base_estimator: BaseEstimator | None = None, clip: float = 0.01):
        super().__init__(base_estimator=base_estimator)
        self.clip = clip

    def _fit(self) -> None:
        X = np.vstack([self.X_accept_, self.X_reject_])
        s = np.concatenate(
            [
                np.ones(self.X_accept_.shape[0], dtype=int),
                np.zeros(self.X_reject_.shape[0], dtype=int),
            ]
        )
        self.selection_model_ = self._make_base()
        self.selection_model_.fit(X, s)

    def resample(self) -> ArrayTriple:
        p_accept = self.selection_model_.predict_proba(self.X_accept_)[:, 1]
        p_accept = np.clip(p_accept, self.clip, 1.0)
        w = 1.0 / p_accept
        w = w * (w.shape[0] / w.sum())  # normalise mean weight to 1
        return self.X_accept_.copy(), self.y_accept_.copy(), w
