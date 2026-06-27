# rejectkit

**Reject inference for credit scoring — scikit-learn-compatible methods, plus an honest benchmark that tells you whether reject inference actually helps on your data.**

A credit model is trained on *accepted* applicants, whose good/bad outcome you observe. But it must score the *whole* through-the-door population, including the applicants you *rejected*, who never get an outcome. Training on accepts only is textbook sample-selection bias. **Reject inference** is the family of techniques that corrects it — standard in credit risk, yet missing from the Python ecosystem until now.

## Install

```bash
pip install -e .              # core
pip install -e ".[plot]"      # + matplotlib plots
pip install -e ".[polars]"    # + polars input support
```

Core dependencies: `numpy`, `pandas`, `scikit-learn`.

## 30-second example

```python
from sklearn.linear_model import LogisticRegression
from rejectkit import RejectInferenceClassifier

clf = RejectInferenceClassifier(
    estimator=LogisticRegression(max_iter=1000),
    method="parcelling",
    method_params={"uplift": 1.3},
)
clf.fit(X_accept, y_accept, X_reject)   # rejects have no labels
pd_bad = clf.predict_proba(X_new)[:, 1]
```

See **Methods** for the full catalogue, **Benchmark** for how to test whether reject inference helps, and **Diagnostics & plotting** for the supporting tools.
