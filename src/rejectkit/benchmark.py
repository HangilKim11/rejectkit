"""Benchmark harness for reject inference methods.

The core difficulty of reject inference is that you never observe the outcome
of rejected applicants, so you cannot directly validate any method on real
data. :class:`MaskedRejectBenchmark` sidesteps this on a *fully labelled*
dataset: it hides the labels of a synthetically "rejected" subset of the
training data, asks each method to recover a model, and scores every model on
an untouched test set representing the true through-the-door population.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ._compat import to_numpy_1d, to_numpy_2d
from .diagnostics import auc, gini, ks_statistic
from .estimator import RejectInferenceClassifier
from .methods import HeckmanClassifier


class MaskedRejectBenchmark:
    """Compare reject inference methods by masking labels on labelled data.

    Parameters
    ----------
    selection : {'mar', 'mnar', 'cutoff'}, default='mnar'
        Acceptance mechanism applied to the training set.
        ``'mar'`` — acceptance depends only on observed features.
        ``'mnar'`` — acceptance also depends on the hidden outcome.
        ``'cutoff'`` — accept the lowest-PD fraction under a quick model, the
        realistic credit-policy mechanism (strong, score-based truncation).
    accept_rate : float, default=0.6
        Fraction of training applicants accepted.
    test_size : float, default=0.3
        Fraction held out as the unbiased evaluation population.
    selection_strength : float, default=2.0
        Strength of the feature (and, under MNAR, outcome) dependence. Unused
        for ``'cutoff'``.
    random_state : int, default=0
    """

    def __init__(self, selection="mnar", accept_rate=0.6, test_size=0.3,
                 selection_strength=2.0, random_state=0):
        self.selection = selection
        self.accept_rate = accept_rate
        self.test_size = test_size
        self.selection_strength = selection_strength
        self.random_state = random_state

    def _selection_scores(self, X_std, y, rng):
        n, d = X_std.shape
        w = rng.normal(size=d)
        feat = X_std @ w
        feat = feat / (feat.std() + 1e-12)
        noise = rng.normal(size=n)
        if self.selection == "mar":
            return self.selection_strength * feat + noise
        if self.selection == "mnar":
            y_term = self.selection_strength * (0.5 - y) * 2.0
            return feat + y_term + noise
        raise ValueError("selection must be 'mar', 'mnar', or 'cutoff'.")

    def _accept_mask(self, X_train, y_train) -> np.ndarray:
        if self.selection == "cutoff":
            risk = (
                LogisticRegression(max_iter=1000)
                .fit(X_train, y_train)
                .predict_proba(X_train)[:, 1]
            )
            accepted = risk <= np.quantile(risk, self.accept_rate)
        else:
            rng = np.random.default_rng(self.random_state)
            X_std = StandardScaler().fit_transform(X_train)
            z = self._selection_scores(X_std, y_train, rng)
            accepted = z >= np.quantile(z, 1.0 - self.accept_rate)
        if len(np.unique(y_train[accepted])) < 2:
            raise ValueError(
                "The accepted subset contains a single class; lower "
                "selection_strength or change accept_rate."
            )
        return accepted

    def compare(self, methods: Sequence[str], X, y, estimator=None,
                method_params: dict | None = None) -> pd.DataFrame:
        """Run the benchmark and return a tidy results table.

        Methods may include ``'heckman'`` (uses :class:`HeckmanClassifier`) as
        well as any resampling method name. Returns a DataFrame indexed by
        ``['oracle', 'naive', *methods]`` with columns ``auc``, ``ks``, ``gini``
        and ``auc_recovery`` (0 = no better than naive, 1 = matches oracle).
        Sample sizes are in ``df.attrs``.
        """
        X = to_numpy_2d(X)
        y = to_numpy_1d(y).astype(int)
        method_params = method_params or {}
        est = estimator if estimator is not None else LogisticRegression(max_iter=1000)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        accepted = self._accept_mask(X_tr, y_tr)
        X_a, y_a, X_r = X_tr[accepted], y_tr[accepted], X_tr[~accepted]

        def _scores(model):
            p = model.predict_proba(X_te)[:, 1]
            return {"auc": auc(y_te, p), "ks": ks_statistic(y_te, p), "gini": gini(y_te, p)}

        rows = {"oracle": _scores(clone(est).fit(X_tr, y_tr)),
                "naive": _scores(clone(est).fit(X_a, y_a))}
        for m in methods:
            if str(m).lower() == "heckman":
                clf = HeckmanClassifier(outcome_estimator=clone(est))
            else:
                clf = RejectInferenceClassifier(
                    estimator=clone(est), method=m, method_params=method_params.get(m, {})
                )
            clf.fit(X_a, y_a, X_r)
            rows[m] = _scores(clf)

        df = pd.DataFrame(rows).T[["auc", "ks", "gini"]]
        gap = rows["oracle"]["auc"] - rows["naive"]["auc"]
        # Recovery is only meaningful when the oracle clearly beats the naive
        # model; if there is no gap to recover, report NaN rather than a ratio
        # that explodes around a near-zero denominator.
        df["auc_recovery"] = (
            (df["auc"] - rows["naive"]["auc"]) / gap if gap > 5e-3 else np.nan
        )
        df.attrs.update(
            n_accept=int(accepted.sum()),
            n_reject=int((~accepted).sum()),
            n_test=int(len(y_te)),
            selection=self.selection,
        )
        return df
