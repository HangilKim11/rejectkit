import numpy as np
import pytest

from rejectkit import MaskedRejectBenchmark
from rejectkit.datasets import make_credit_data

METHODS = ["fuzzy", "parcelling", "reweighting", "reclassification",
           "extrapolation", "selflearning", "heckman"]


@pytest.fixture(scope="module")
def labelled():
    return make_credit_data(n_samples=1800, random_state=2)


@pytest.mark.parametrize("selection", ["mar", "mnar", "cutoff"])
def test_compare_output_shape(labelled, selection):
    X, y = labelled
    df = MaskedRejectBenchmark(
        selection=selection, selection_strength=1.5, random_state=0
    ).compare(METHODS, X, y)
    assert list(df.index) == ["oracle", "naive", *METHODS]
    assert {"auc", "ks", "gini", "auc_recovery"}.issubset(df.columns)
    assert df["auc"].between(0.0, 1.0).all()
    assert df.attrs["n_accept"] > 0 and df.attrs["n_reject"] > 0


def test_oracle_not_worse_than_naive(labelled):
    X, y = labelled
    df = MaskedRejectBenchmark(selection="mnar", selection_strength=1.5, random_state=0).compare(
        METHODS, X, y
    )
    assert df.loc["oracle", "auc"] >= df.loc["naive", "auc"] - 0.02
    assert np.isfinite(df["auc_recovery"]).all()


def test_invalid_selection_raises(labelled):
    X, y = labelled
    with pytest.raises(ValueError):
        MaskedRejectBenchmark(selection="nope").compare(["fuzzy"], X, y)
