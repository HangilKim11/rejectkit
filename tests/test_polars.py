import pytest

pl = pytest.importorskip("polars")

from rejectkit import MaskedRejectBenchmark, RejectInferenceClassifier
from rejectkit.datasets import make_accept_reject, make_credit_data


def _to_polars_df(pdf):
    return pl.DataFrame({c: pdf[c].to_numpy() for c in pdf.columns})


def test_polars_classifier():
    Xa, ya, Xr, _ = make_accept_reject(n_samples=1500, selection_strength=1.2, random_state=0)
    clf = RejectInferenceClassifier(method="fuzzy").fit(
        _to_polars_df(Xa), pl.Series("bad", ya.to_numpy()), _to_polars_df(Xr)
    )
    assert clf.predict_proba(_to_polars_df(Xr)).shape == (len(Xr), 2)


def test_polars_benchmark():
    X, y = make_credit_data(n_samples=1500, random_state=0)
    df = MaskedRejectBenchmark(selection="mar", selection_strength=1.5, random_state=0).compare(
        ["fuzzy"], _to_polars_df(X), pl.Series("bad", y.to_numpy())
    )
    assert df["auc"].between(0, 1).all()
