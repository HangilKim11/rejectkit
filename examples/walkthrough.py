"""rejectkit — function-by-function walkthrough on sample data.

Run:  pip install -e ".[plot]"  &&  python examples/walkthrough.py
Every public piece of the library is exercised below with small, readable data.
"""

import matplotlib
matplotlib.use("Agg")  # headless: save figures to files instead of showing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from rejectkit import (
    Extrapolation,
    FuzzyAugmentation,
    HeckmanClassifier,
    MaskedRejectBenchmark,
    Parcelling,
    Reclassification,
    RejectInferenceClassifier,
    Reweighting,
    SelfLearning,
    SimpleAugmentation,
    get_inferencer,
    plotting,
)
from rejectkit.datasets import make_accept_reject, make_credit_data
from rejectkit.diagnostics import auc, feature_drift, gini, ks_statistic, psi, swap_set

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)


def hr(title):
    print("\n" + "=" * 78 + f"\n# {title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
hr("1) datasets.make_credit_data  — a fully labelled toy credit dataset")
X, y = make_credit_data(n_samples=2000, n_features=6, bad_rate=0.3, random_state=0)
print("X shape:", X.shape, "| y shape:", y.shape, "| overall bad rate:", round(y.mean(), 3))
print("X.head(3):")
print(X.head(3).round(3))
print("y.head(8):", y.head(8).tolist())

hr("2) datasets.make_accept_reject  — accepts (labelled) + rejects (no label)")
X_accept, y_accept, X_reject, y_reject_true = make_accept_reject(
    n_samples=2000, n_features=6, accept_rate=0.6, random_state=0
)
print("X_accept:", X_accept.shape, "| y_accept bad rate:", round(y_accept.mean(), 3))
print("X_reject:", X_reject.shape, "| (y_reject_true is hidden in practice; bad rate:",
      round(y_reject_true.mean(), 3), ")")
print("X_accept.head(3):")
print(X_accept.head(3).round(3))

# ----------------------------------------------------------------------------
hr("3) Reject inference methods — fit_resample() output shapes & weights")
resamplers = {
    "SimpleAugmentation": SimpleAugmentation(),
    "FuzzyAugmentation": FuzzyAugmentation(),
    "Parcelling(uplift=1.3)": Parcelling(uplift=1.3),
    "Reclassification": Reclassification(),
    "Extrapolation": Extrapolation(n_neighbors=15),
    "Reweighting": Reweighting(),
    "SelfLearning": SelfLearning(),
}
print(f"{'method':24s} {'rows':>6} {'=accepts+extra':>16} {'weight min/mean/max':>26}")
for name, m in resamplers.items():
    Xn, yn, w = m.fit_resample(X_accept, y_accept, X_reject)
    extra = len(Xn) - len(X_accept)
    print(f"{name:24s} {len(Xn):>6} {f'{len(X_accept)}+{extra}':>16} "
          f"{f'{w.min():.2f}/{w.mean():.2f}/{w.max():.2f}':>26}")

print("\nConcrete look at FuzzyAugmentation for reject #0:")
Xf, yf, wf = FuzzyAugmentation().fit_resample(X_accept, y_accept, X_reject)
na, nr = len(X_accept), len(X_reject)
print(f"  reject #0 becomes 2 rows -> label={yf[na]} weight={wf[na]:.3f}  and  "
      f"label={yf[na+nr]} weight={wf[na+nr]:.3f}  (sum={wf[na]+wf[na+nr]:.3f})")

# ----------------------------------------------------------------------------
hr("4) get_inferencer — build a method by name")
inf = get_inferencer("fuzzy")
print("get_inferencer('fuzzy') ->", type(inf).__name__)
print("get_inferencer('twins') ->", type(get_inferencer("twins")).__name__, "(alias of Extrapolation)")

# ----------------------------------------------------------------------------
hr("5) RejectInferenceClassifier — the one-click wrapper")
clf = RejectInferenceClassifier(
    estimator=LogisticRegression(max_iter=1000),
    method="parcelling",
    method_params={"uplift": 1.2},
).fit(X_accept, y_accept, X_reject)
proba = clf.predict_proba(X_reject)
print("predict_proba(X_reject) shape:", proba.shape, "(col 0 = P(good), col 1 = P(bad))")
print("first 3 rows:\n", proba[:3].round(3))
print("predict(X_reject)[:12]:", clf.predict(X_reject)[:12])
print("P(bad) for first 5 rejects:", proba[:5, 1].round(3))

# swap the inner model for a RandomForest — same API
rf = RejectInferenceClassifier(
    estimator=RandomForestClassifier(n_estimators=50, random_state=0), method="fuzzy"
).fit(X_accept, y_accept, X_reject)
print("with RandomForest, P(bad) first 5:", rf.predict_proba(X_reject)[:5, 1].round(3))

# ----------------------------------------------------------------------------
hr("6) HeckmanClassifier — standalone two-step control-function model")
heck = HeckmanClassifier().fit(X_accept, y_accept, X_reject)
print("predict_proba(X_reject) first 3:\n", heck.predict_proba(X_reject)[:3].round(3))

# ----------------------------------------------------------------------------
hr("7) MaskedRejectBenchmark — does reject inference help on this data?")
bench = MaskedRejectBenchmark(selection="mnar", accept_rate=0.6, random_state=0)
results = bench.compare(
    ["simple", "fuzzy", "parcelling", "reweighting", "extrapolation", "selflearning", "heckman"],
    X, y,
)
print(results.round(4).to_string())
print("attrs:", {k: results.attrs[k] for k in ("n_accept", "n_reject", "n_test", "selection")})

# ----------------------------------------------------------------------------
hr("8) diagnostics — auc / gini / ks_statistic / psi")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
score = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
print("auc         :", round(auc(yte, score), 4))
print("gini        :", round(gini(yte, score), 4))
print("ks_statistic:", round(ks_statistic(yte, score), 4))
print("psi(train scores vs test scores):",
      round(psi(LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xtr)[:, 1], score), 4))

hr("8b) diagnostics.feature_drift — per-feature accept-vs-reject PSI")
print(feature_drift(X_accept, X_reject).round(3).to_string())

hr("8c) diagnostics.swap_set — reference vs challenger scorecard")
m_ref = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
m_new = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xtr, ytr)
s_ref = m_ref.predict_proba(Xte)[:, 1]
s_new = m_new.predict_proba(Xte)[:, 1]
cut = np.quantile(s_ref, 0.6)  # accept the safest 60% under each model
print(swap_set(yte, s_ref, s_new, cut, np.quantile(s_new, 0.6)).round(3).to_string())

# ----------------------------------------------------------------------------
hr("9) plotting — saved to examples/plots/*.png")
ax = plotting.plot_benchmark(results)
ax.figure.tight_layout(); ax.figure.savefig("examples/plots/benchmark.png", dpi=110)
ax = plotting.plot_score_distributions(clf.predict_proba(X_accept)[:, 1], clf.predict_proba(X_reject)[:, 1])
ax.figure.tight_layout(); ax.figure.savefig("examples/plots/score_distributions.png", dpi=110)
ax = plotting.plot_ks(yte, score)
ax.figure.tight_layout(); ax.figure.savefig("examples/plots/ks_curve.png", dpi=110)
print("saved: benchmark.png, score_distributions.png, ks_curve.png")
print("\nDONE.")
