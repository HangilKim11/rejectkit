"""Parcelling (score-band) reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from ..base import ArrayTriple, BaseRejectInferencer


class Parcelling(BaseRejectInferencer):
    """Parcelling.

    Applicants are scored and split into ``n_bins`` score bands (quantiles of
    the accept score distribution). Within each band the observed accept bad
    rate is multiplied by ``uplift`` (>= 1) — encoding the assumption that
    rejects are worse risks than accepts with the same score — and used as the
    reject bad rate for that band.

    With ``assignment='expected'`` (default) each reject contributes two
    deterministically weighted rows (no randomness); with ``'random'`` a label
    is drawn from a Bernoulli with the band's reject bad rate.

    Parameters
    ----------
    n_bins : int, default=10
        Number of score bands.
    uplift : float, default=1.0
        Multiplier applied to each band's accept bad rate. ``1.0`` assumes
        rejects behave like accepts of the same score; ``> 1`` makes them worse.
    assignment : {'expected', 'random'}, default='expected'
    random_state : int, optional
        Used only when ``assignment='random'``.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        n_bins: int = 10,
        uplift: float = 1.0,
        assignment: str = "expected",
        random_state: int | None = None,
    ):
        super().__init__(base_estimator=base_estimator)
        self.n_bins = n_bins
        self.uplift = uplift
        self.assignment = assignment
        self.random_state = random_state

    def _reject_bad_rate(self) -> np.ndarray:
        accept_score = self.scorer_.predict_proba(self.X_accept_)[:, 1]
        reject_score = self._reject_bad_proba()
        edges = np.unique(np.quantile(accept_score, np.linspace(0, 1, self.n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        inner = edges[1:-1]
        a_bin = np.digitize(accept_score, inner)
        r_bin = np.digitize(reject_score, inner)
        n_eff = len(edges) - 1
        bad_rate = np.full(n_eff, float(self.y_accept_.mean()))
        for b in range(n_eff):
            mask = a_bin == b
            if mask.any():
                bad_rate[b] = self.y_accept_[mask].mean()
        return np.clip(bad_rate[r_bin] * self.uplift, 0.0, 1.0)

    def resample(self) -> ArrayTriple:
        reject_bad_rate = self._reject_bad_rate()

        if self.assignment == "random":
            rng = np.random.default_rng(self.random_state)
            y_reject = (rng.random(reject_bad_rate.shape[0]) < reject_bad_rate).astype(int)
            X = np.vstack([self.X_accept_, self.X_reject_])
            y = np.concatenate([self.y_accept_, y_reject])
            w = np.ones(X.shape[0])
            return X, y, w

        if self.assignment == "expected":
            n_a = self.X_accept_.shape[0]
            n_r = self.X_reject_.shape[0]
            X = np.vstack([self.X_accept_, self.X_reject_, self.X_reject_])
            y = np.concatenate(
                [self.y_accept_, np.ones(n_r, dtype=int), np.zeros(n_r, dtype=int)]
            )
            w = np.concatenate([np.ones(n_a), reject_bad_rate, 1.0 - reject_bad_rate])
            return X, y, w

        raise ValueError("assignment must be 'expected' or 'random'.")
