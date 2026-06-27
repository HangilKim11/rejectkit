"""Nearest-neighbour extrapolation ('twins') reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..base import ArrayTriple, BaseRejectInferencer


class Extrapolation(BaseRejectInferencer):
    """Nearest-neighbour label extrapolation (a.k.a. 'twins').

    Each reject is matched to its ``n_neighbors`` most similar accepts in
    standardised feature space; the local bad rate among those neighbours
    becomes the reject's P(bad). As in fuzzy augmentation, each reject then
    contributes two weighted rows. No parametric model is assumed for the
    reject labels.

    Parameters
    ----------
    n_neighbors : int, default=10
    """

    def __init__(self, base_estimator: BaseEstimator | None = None,
                 n_neighbors: int = 10):
        super().__init__(base_estimator=base_estimator)
        self.n_neighbors = n_neighbors

    def _fit(self) -> None:
        self.scaler_ = StandardScaler().fit(self.X_accept_)
        k = min(self.n_neighbors, self.X_accept_.shape[0])
        self.nn_ = NearestNeighbors(n_neighbors=k).fit(
            self.scaler_.transform(self.X_accept_)
        )

    def resample(self) -> ArrayTriple:
        Xa, ya, Xr = self.X_accept_, self.y_accept_, self.X_reject_
        idx = self.nn_.kneighbors(self.scaler_.transform(Xr), return_distance=False)
        p_bad = ya[idx].mean(axis=1).astype(float)
        n_a, n_r = Xa.shape[0], Xr.shape[0]
        X = np.vstack([Xa, Xr, Xr])
        y = np.concatenate([ya, np.ones(n_r, dtype=int), np.zeros(n_r, dtype=int)])
        w = np.concatenate([np.ones(n_a), p_bad, 1.0 - p_bad])
        return X, y, w
