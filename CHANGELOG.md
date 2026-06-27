# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-06-27

### Added
- `SelfLearning` — semi-supervised self-training reject inference.
- `HeckmanClassifier` — two-step control-function correction (inverse Mills ratio).
- **Polars** support: all public entry points accept pandas, polars, or numpy.
- `rejectkit.plotting` — `plot_benchmark`, `plot_score_distributions`, `plot_ks`
  (optional, `pip install rejectkit[plot]`).
- `diagnostics.feature_drift` (per-feature accept-vs-reject PSI) and
  `diagnostics.swap_set` (swap-set analysis between two scorecards).
- `MaskedRejectBenchmark` gains `selection="cutoff"` (realistic PD-cutoff policy)
  and can benchmark `"heckman"` alongside the resampling methods.
- Documentation site (`mkdocs` + Material).

### Changed
- `auc_recovery` now reports `NaN` when the oracle does not clearly beat the
  naive model, instead of an unstable ratio around a near-zero denominator.

## [0.2.0] - 2026-06-27

### Added
- `Reclassification` — iterative relabel-and-refit reject inference.
- `Extrapolation` (alias `twins`) — nearest-neighbour label extrapolation.

## [0.1.0] - 2026-06-26

### Added
- `BaseRejectInferencer` and the scikit-learn-style `fit` / `resample` /
  `fit_resample` API.
- Methods: `SimpleAugmentation`, `FuzzyAugmentation`, `Parcelling`, `Reweighting`.
- `RejectInferenceClassifier` — wrap any classifier with reject inference.
- `MaskedRejectBenchmark` — measure whether reject inference helps on your data.
- Metrics: `auc`, `gini`, `ks_statistic`, `psi`.
- Synthetic data generators `make_credit_data`, `make_accept_reject`.
