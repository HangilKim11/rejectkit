import numpy as np
import pytest

from rejectkit import (
    Extrapolation,
    FuzzyAugmentation,
    Parcelling,
    Reclassification,
    Reweighting,
    SelfLearning,
    SimpleAugmentation,
)
from rejectkit.datasets import make_accept_reject

RESAMPLERS = [
    SimpleAugmentation, FuzzyAugmentation, Parcelling, Reweighting,
    Reclassification, Extrapolation, SelfLearning,
]


@pytest.fixture(scope="module")
def data():
    Xa, ya, Xr, _ = make_accept_reject(n_samples=1500, selection_strength=1.2, random_state=0)
    assert set(np.unique(ya)) == {0, 1}
    return Xa, ya, Xr


@pytest.mark.parametrize("Method", RESAMPLERS)
def test_resample_contract(Method, data):
    Xa, ya, Xr = data
    X, y, w = Method().fit_resample(Xa, ya, Xr)
    assert len(X) == len(y) == len(w)
    assert np.all(np.isfinite(w)) and np.all(w >= 0)
    assert set(np.unique(y)).issubset({0, 1})
    assert X.shape[1] == Xa.shape[1]
    assert len(X) >= len(Xa)  # accepts are always retained


def test_fuzzy_doubles_rejects(data):
    Xa, ya, Xr = data
    X, y, w = FuzzyAugmentation().fit_resample(Xa, ya, Xr)
    assert len(X) == len(Xa) + 2 * len(Xr)
    n_a, n_r = len(Xa), len(Xr)
    np.testing.assert_allclose(w[n_a:n_a + n_r] + w[n_a + n_r:], 1.0)


def test_parcelling_expected_vs_random(data):
    Xa, ya, Xr = data
    Xe, _, _ = Parcelling(assignment="expected").fit_resample(Xa, ya, Xr)
    assert len(Xe) == len(Xa) + 2 * len(Xr)
    Xr2, _, _ = Parcelling(assignment="random", random_state=0).fit_resample(Xa, ya, Xr)
    assert len(Xr2) == len(Xa) + len(Xr)


def test_parcelling_uplift_increases_bad_rate(data):
    Xa, ya, Xr = data
    base = Parcelling(uplift=1.0).fit(Xa, ya, Xr)._reject_bad_rate().mean()
    up = Parcelling(uplift=1.5).fit(Xa, ya, Xr)._reject_bad_rate().mean()
    assert up >= base


def test_reweighting_keeps_accepts_only(data):
    Xa, ya, Xr = data
    X, _, w = Reweighting().fit_resample(Xa, ya, Xr)
    assert len(X) == len(Xa)
    np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-6)


def test_extrapolation_doubles_rejects(data):
    Xa, ya, Xr = data
    X, _, _ = Extrapolation(n_neighbors=15).fit_resample(Xa, ya, Xr)
    assert len(X) == len(Xa) + 2 * len(Xr)


def test_selflearning_subset(data):
    Xa, ya, Xr = data
    X, y, w = SelfLearning(threshold=0.8).fit_resample(Xa, ya, Xr)
    assert len(Xa) <= len(X) <= len(Xa) + len(Xr)
    np.testing.assert_allclose(w, 1.0)


def test_validation_errors(data):
    Xa, ya, Xr = data
    with pytest.raises(ValueError):
        FuzzyAugmentation().fit(Xa, ya, Xr.iloc[:, :-1])
    with pytest.raises(ValueError):
        FuzzyAugmentation().fit(Xa, np.zeros(len(ya)), Xr)
