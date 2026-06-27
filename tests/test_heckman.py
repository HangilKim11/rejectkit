import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from rejectkit import HeckmanClassifier
from rejectkit.datasets import make_accept_reject


@pytest.fixture(scope="module")
def data():
    Xa, ya, Xr, _ = make_accept_reject(n_samples=1500, selection_strength=1.2, random_state=2)
    return Xa, ya, Xr


def test_fit_predict(data):
    Xa, ya, Xr = data
    clf = HeckmanClassifier().fit(Xa, ya, Xr)
    proba = clf.predict_proba(Xr)
    assert proba.shape == (len(Xr), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert set(np.unique(clf.predict(Xr))).issubset({0, 1})


def test_custom_estimators(data):
    Xa, ya, Xr = data
    clf = HeckmanClassifier(
        outcome_estimator=RandomForestClassifier(n_estimators=20, random_state=0)
    ).fit(Xa, ya, Xr)
    assert clf.predict_proba(Xr).shape == (len(Xr), 2)
