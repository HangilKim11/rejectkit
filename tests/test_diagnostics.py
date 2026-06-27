import numpy as np

from rejectkit.datasets import make_accept_reject
from rejectkit.diagnostics import auc, feature_drift, gini, ks_statistic, psi, swap_set


def test_auc_gini_ks_ranges():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    score = rng.random(500)
    assert 0.0 <= auc(y, score) <= 1.0
    assert -1.0 <= gini(y, score) <= 1.0
    assert 0.0 <= ks_statistic(y, score) <= 1.0


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, score) == 1.0
    assert ks_statistic(y, score) == 1.0


def test_psi_zero_for_identical():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(2)
    assert psi(rng.normal(size=2000), rng.normal(loc=1.0, size=2000)) > 0.1


def test_feature_drift_shape():
    Xa, _, Xr, _ = make_accept_reject(n_samples=1500, random_state=0)
    fd = feature_drift(Xa, Xr)
    assert len(fd) == Xa.shape[1]
    assert (fd >= 0).all()
    assert set(fd.index) == set(Xa.columns)  # sorted by PSI desc


def test_swap_set():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=400)
    s_ref, s_new = rng.random(400), rng.random(400)
    out = swap_set(y, s_ref, s_new, 0.5, 0.5)
    assert list(out.index) == ["kept_accept", "swap_out", "swap_in", "kept_reject"]
    assert out["n"].sum() == 400
