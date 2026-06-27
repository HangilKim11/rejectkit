# API reference

## Estimators

- **`RejectInferenceClassifier(estimator=None, method="fuzzy", base_scorer=None, method_params=None)`**
  Wrap any classifier with reject inference. `fit(X_accept, y_accept, X_reject)`, then `predict` / `predict_proba`.
- **`HeckmanClassifier(selection_estimator=None, outcome_estimator=None)`**
  Two-step control-function correction. `fit(X_accept, y_accept, X_reject)`, then `predict_proba`.
- **`get_inferencer(method, base_estimator=None, **params)`** — factory for the resampler classes.

## Reject inference methods (resamplers)

All subclass `BaseRejectInferencer` and expose `fit`, `resample`, `fit_resample`:

`SimpleAugmentation`, `FuzzyAugmentation`, `Parcelling`, `Reclassification`,
`Extrapolation`, `Reweighting`, `SelfLearning`.

## Benchmark

- **`MaskedRejectBenchmark(selection="mnar", accept_rate=0.6, test_size=0.3, selection_strength=2.0, random_state=0)`**
  `.compare(methods, X, y, estimator=None, method_params=None) -> DataFrame`.

## Data & diagnostics

- `datasets.make_credit_data(...)`, `datasets.make_accept_reject(...)`
- `diagnostics.auc / gini / ks_statistic / psi / feature_drift / swap_set`
- `plotting.plot_benchmark / plot_score_distributions / plot_ks`
