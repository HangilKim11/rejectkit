"""Augmentation-based reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ArrayTriple, BaseRejectInferencer


class SimpleAugmentation(BaseRejectInferencer):
    """Hard-cutoff augmentation.

    Scores the rejects with a good/bad model fitted on the accepts and assigns
    a hard 0/1 label by thresholding P(bad). The labelled rejects are appended
    to the accepts with unit weight. Simple and transparent, but sensitive to
    the chosen ``threshold`` and prone to over-confidence.

    Parameters
    ----------
    threshold : float, default=0.5
        Rejects with ``P(bad) >= threshold`` are labelled bad (1).
    """

    def __init__(self, base_estimator: BaseEstimator | None = None, threshold: float = 0.5):
        super().__init__(base_estimator=base_estimator)
        self.threshold = threshold

    def resample(self) -> ArrayTriple:
        p_bad = self._reject_bad_proba()
        y_reject = (p_bad >= self.threshold).astype(int)
        X = np.vstack([self.X_accept_, self.X_reject_])
        y = np.concatenate([self.y_accept_, y_reject])
        w = np.ones(X.shape[0])
        return X, y, w


class FuzzyAugmentation(BaseRejectInferencer):
    """Fuzzy augmentation (a.k.a. fuzzy parcelling).

    Instead of committing to a hard label, each reject contributes **two** rows
    — one labelled bad, one labelled good — weighted by the model's P(bad) and
    P(good). This avoids the over-confidence of hard cutoffs and is generally
    the most stable augmentation method.

    Parameters
    ----------
    reject_weight : float, default=1.0
        Global multiplier on the weight of reject-derived rows, useful to
        reflect the reject share of the through-the-door population.
    """

    def __init__(self, base_estimator: BaseEstimator | None = None, reject_weight: float = 1.0):
        super().__init__(base_estimator=base_estimator)
        self.reject_weight = reject_weight

    def resample(self) -> ArrayTriple:
        p_bad = self._reject_bad_proba()
        p_good = 1.0 - p_bad
        n_a = self.X_accept_.shape[0]
        n_r = self.X_reject_.shape[0]
        X = np.vstack([self.X_accept_, self.X_reject_, self.X_reject_])
        y = np.concatenate(
            [self.y_accept_, np.ones(n_r, dtype=int), np.zeros(n_r, dtype=int)]
        )
        w = np.concatenate(
            [np.ones(n_a), self.reject_weight * p_bad, self.reject_weight * p_good]
        )
        return X, y, w
