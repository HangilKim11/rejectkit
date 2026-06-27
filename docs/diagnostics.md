# Diagnostics & plotting

## Metrics (`rejectkit.diagnostics`)

```python
from rejectkit.diagnostics import auc, gini, ks_statistic, psi, feature_drift, swap_set
```

- `auc`, `gini`, `ks_statistic` — ranking metrics (`y_score` = P(bad)).
- `psi(expected, actual)` — Population Stability Index between two distributions.
- `feature_drift(X_accept, X_reject)` — per-feature PSI between accepts and rejects, sorted worst-first. A quick read on how unrepresentative your accepts are.
- `swap_set(y, score_ref, score_new, cutoff_ref, cutoff_new)` — who a challenger scorecard swaps in/out versus a reference, with bad rates per group.

## Plotting (`rejectkit.plotting`, needs `[plot]`)

```python
from rejectkit import plotting
plotting.plot_benchmark(results)                 # bar chart of auc_recovery
plotting.plot_score_distributions(s_acc, s_rej)  # accept vs reject overlap
plotting.plot_ks(y_true, y_score)                # KS curve
```

Matplotlib is imported lazily, so importing `rejectkit.plotting` is cheap even without matplotlib installed.
