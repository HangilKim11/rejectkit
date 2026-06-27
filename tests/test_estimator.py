import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from rejectkit import RejectInferenceClassifier, get_inferencer
from rejectkit.datasets import make_accept_reject

ALL_METHODS = [
    "simple", "fuzzy", "parcelling", "reweighting",
    "reclassification", "extrapolation", "twins", "selflearning", "self-learning",
]


@pytest.fixture(scope="module")
def data():
    Xa, ya, Xr, _ = make_accept_reject(n_samples=1500, selection_strength=1.2, random_state=1)
    return Xa, ya, Xr


@pytest.mark.parametrize("method", ALL_METHODS)
def test_fit_predict(method, data):
    Xa, ya, Xr = data
    clf = RejectInferenceClassifier(method=method).fit(Xa, ya, Xr)
    proba = clf.predict_proba(Xr)
    assert proba.shape == (len(Xr), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert set(np.unique(clf.predict(Xr))).issubset({0, 1})


def test_custom_estimator(data):
    Xa, ya, Xr = data
    clf = RejectInferenceClassifier(
        estimator=RandomForestClassifier(n_estimators=25, random_state=0),
        method="parcelling", method_params={"uplift": 1.3},
    ).fit(Xa, ya, Xr)
    assert clf.predict_proba(Xr).shape == (len(Xr), 2)


def test_clone_is_sklearn_compatible():
    clf = RejectInferenceClassifier(method="fuzzy", method_params={"reject_weight": 2.0})
    cloned = clone(clf)
    assert cloned.get_params()["method"] == "fuzzy"
    assert cloned.method_params == {"reject_weight": 2.0}


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        get_inferencer("does-not-exist")
