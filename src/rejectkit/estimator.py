"""High-level estimator that wraps any classifier with reject inference."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from ._compat import to_numpy_2d
from .methods import (
    Extrapolation,
    FuzzyAugmentation,
    Parcelling,
    Reclassification,
    Reweighting,
    SelfLearning,
    SimpleAugmentation,
)

_METHODS = {
    "simple": SimpleAugmentation,
    "fuzzy": FuzzyAugmentation,
    "parcelling": Parcelling,
    "reweighting": Reweighting,
    "reclassification": Reclassification,
    "extrapolation": Extrapolation,
    "twins": Extrapolation,
    "selflearning": SelfLearning,
    "self-learning": SelfLearning,
}


def get_inferencer(method: str, base_estimator: BaseEstimator | None = None, **params):
    """Build a reject inferencer by name.

    Parameters
    ----------
    method : str
        One of: simple, fuzzy, parcelling, reweighting, reclassification,
        extrapolation (alias twins), selflearning.
    base_estimator : sklearn classifier, optional
    **params
        Extra keyword arguments for the chosen method (e.g. ``uplift=1.5``).
    """
    key = str(method).lower()
    if key not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Available: {sorted(_METHODS)}.")
    return _METHODS[key](base_estimator=base_estimator, **params)


class RejectInferenceClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a scikit-learn classifier with reject inference.

    Infers labels/weights for the rejected applicants using ``method``, builds
    the augmented through-the-door sample, and fits ``estimator`` on it
    (passing ``sample_weight``). It then behaves like an ordinary fitted
    classifier via :meth:`predict` / :meth:`predict_proba`.

    Note the non-standard ``fit`` signature: it takes accepts and rejects
    separately. Accepts pandas, polars, or numpy inputs.

    Parameters
    ----------
    estimator : sklearn classifier, optional
        Final model trained on the augmented sample. Must accept
        ``sample_weight``. Defaults to LogisticRegression.
    method : str, default='fuzzy'
        See :func:`get_inferencer`.
    base_scorer : sklearn classifier, optional
        Internal model used by the reject inference method.
    method_params : dict, optional
        Extra keyword arguments forwarded to the method, e.g. ``{'uplift': 1.5}``.
    """

    def __init__(self, estimator=None, method="fuzzy", base_scorer=None, method_params=None):
        self.estimator = estimator
        self.method = method
        self.base_scorer = base_scorer
        self.method_params = method_params

    def fit(self, X_accept, y_accept, X_reject):
        params = self.method_params or {}
        inferencer = get_inferencer(self.method, base_estimator=self.base_scorer, **params)
        X_aug, y_aug, w_aug = inferencer.fit_resample(X_accept, y_accept, X_reject)
        self.inferencer_ = inferencer
        self.estimator_ = (
            clone(self.estimator) if self.estimator is not None
            else LogisticRegression(max_iter=1000)
        )
        self.estimator_.fit(X_aug, y_aug, sample_weight=w_aug)
        self.classes_ = getattr(self.estimator_, "classes_", np.array([0, 1]))
        self.n_features_in_ = X_aug.shape[1]
        return self

    def predict(self, X):
        return self.estimator_.predict(to_numpy_2d(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(to_numpy_2d(X))

    def decision_function(self, X):
        Xa = to_numpy_2d(X)
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(Xa)
        return self.predict_proba(Xa)[:, 1]
