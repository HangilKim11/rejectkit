"""Heckman-style control-function correction for sample selection."""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from .._compat import to_numpy_1d, to_numpy_2d

_erf = np.vectorize(math.erf)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


class HeckmanClassifier(BaseEstimator, ClassifierMixin):
    """Heckman-style two-step control-function correction.

    Step 1 fits a *selection* model separating accepts from rejects and derives
    each applicant's inverse Mills ratio (IMR) from its selection score. Step 2
    trains the outcome model on the accepts with the IMR appended as an extra
    feature; the IMR term absorbs the selection bias. At prediction time the
    IMR is recomputed for new applicants and appended in the same way.

    Unlike the resampling methods, Heckman augments the *feature space* rather
    than the sample, so it is a standalone classifier rather than a
    ``BaseRejectInferencer``. The Gaussian-latent assumption is an
    approximation; treat it as a control-function heuristic.

    Parameters
    ----------
    selection_estimator : sklearn classifier, optional
        Separates accepts (1) from rejects (0). Defaults to LogisticRegression.
    outcome_estimator : sklearn classifier, optional
        Trained on accepts + IMR. Defaults to LogisticRegression.
    """

    def __init__(self, selection_estimator: BaseEstimator | None = None,
                 outcome_estimator: BaseEstimator | None = None):
        self.selection_estimator = selection_estimator
        self.outcome_estimator = outcome_estimator

    def _sel(self):
        if self.selection_estimator is not None:
            return clone(self.selection_estimator)
        return LogisticRegression(max_iter=1000)

    def _out(self):
        if self.outcome_estimator is not None:
            return clone(self.outcome_estimator)
        return LogisticRegression(max_iter=1000)

    def _latent(self, X):
        if hasattr(self.selection_, "decision_function"):
            z = self.selection_.decision_function(X)
        else:
            p = np.clip(self.selection_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
        return (np.asarray(z, dtype=float) - self.z_mean_) / self.z_std_

    def _imr(self, X):
        z = self._latent(X)
        return _norm_pdf(z) / np.clip(_norm_cdf(z), 1e-6, None)

    def fit(self, X_accept, y_accept, X_reject):
        Xa, Xr = to_numpy_2d(X_accept), to_numpy_2d(X_reject)
        ya = to_numpy_1d(y_accept).astype(int)
        Xs = np.vstack([Xa, Xr])
        s = np.concatenate([np.ones(Xa.shape[0], dtype=int),
                            np.zeros(Xr.shape[0], dtype=int)])
        self.selection_ = self._sel().fit(Xs, s)
        if hasattr(self.selection_, "decision_function"):
            z_raw = self.selection_.decision_function(Xa)
        else:
            p = np.clip(self.selection_.predict_proba(Xa)[:, 1], 1e-6, 1 - 1e-6)
            z_raw = np.log(p / (1 - p))
        z_raw = np.asarray(z_raw, dtype=float)
        self.z_mean_, self.z_std_ = float(z_raw.mean()), float(z_raw.std() + 1e-12)
        imr = self._imr(Xa)
        self.outcome_ = self._out().fit(np.column_stack([Xa, imr]), ya)
        self.classes_ = getattr(self.outcome_, "classes_", np.array([0, 1]))
        self.n_features_in_ = Xa.shape[1]
        return self

    def predict_proba(self, X):
        X = to_numpy_2d(X)
        return self.outcome_.predict_proba(np.column_stack([X, self._imr(X)]))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
