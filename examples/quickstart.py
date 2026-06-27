"""rejectkit quickstart.

Run from the repo root:

    pip install -e ".[plot]"
    python examples/quickstart.py
"""

from sklearn.linear_model import LogisticRegression

from rejectkit import MaskedRejectBenchmark, RejectInferenceClassifier
from rejectkit.datasets import make_accept_reject, make_credit_data
from rejectkit.diagnostics import feature_drift

ALL_METHODS = [
    "simple", "fuzzy", "parcelling", "reweighting",
    "reclassification", "extrapolation", "selflearning", "heckman",
]


def single_model_demo() -> None:
    """Train one model with reject inference and inspect accept/reject drift."""
    X_accept, y_accept, X_reject, _ = make_accept_reject(random_state=0)
    clf = RejectInferenceClassifier(
        estimator=LogisticRegression(max_iter=1000),
        method="parcelling",
        method_params={"uplift": 1.3},
    )
    clf.fit(X_accept, y_accept, X_reject)
    print("=== Single model (parcelling, uplift=1.3) ===")
    print(f"accepts={len(X_accept)}  rejects={len(X_reject)}")
    print("P(bad) for first 5 rejects:", clf.predict_proba(X_reject)[:5, 1].round(3))
    print("\nTop accept-vs-reject feature drift (PSI):")
    print(feature_drift(X_accept, X_reject).head(3).round(3).to_string())
    print()


def benchmark_demo() -> None:
    """Does reject inference help on this data? Measure it under two regimes."""
    X, y = make_credit_data(n_samples=4000, random_state=0)
    for selection in ["cutoff", "mnar"]:
        res = MaskedRejectBenchmark(
            selection=selection, accept_rate=0.6, random_state=0
        ).compare(ALL_METHODS, X, y)
        print(f"=== Benchmark ({selection} selection) — "
              f"accepts={res.attrs['n_accept']}  rejects={res.attrs['n_reject']}  "
              f"test={res.attrs['n_test']} ===")
        print(res.round(4).to_string())
        print()
    print("auc_recovery: 0 = no better than naive (accepts-only),"
          " 1 = matches the full-data oracle.")


if __name__ == "__main__":
    single_model_demo()
    benchmark_demo()
