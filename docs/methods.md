# Methods

Every resampling method shares one API:

```python
method.fit(X_accept, y_accept, X_reject)   # then
X, y, sample_weight = method.resample()    # augmented through-the-door sample
# or in one call:
X, y, sample_weight = method.fit_resample(X_accept, y_accept, X_reject)
```

Pass any `base_estimator` implementing `predict_proba`. Inputs may be pandas, polars, or numpy. Label convention: `1 = bad` (event), `0 = good`.

| Method | Class | Core idea | Assumption |
|---|---|---|---|
| Simple augmentation | `SimpleAugmentation` | Hard 0/1 label by score cutoff | Accept model ranks rejects |
| Fuzzy augmentation | `FuzzyAugmentation` | Two weighted rows per reject (P(bad), P(good)) | MAR; smooth labels |
| Parcelling | `Parcelling` | Per-score-band bad rate × `uplift` | Rejects worse by a fixed factor |
| Reclassification | `Reclassification` | Iteratively relabel & refit | Labels converge |
| Extrapolation / twins | `Extrapolation` | Local bad rate of nearest accepts | Similar applicants behave alike |
| Inverse-propensity reweighting | `Reweighting` | Reweight accepts by `1/P(accept)` | MAR; no labels invented |
| Self-training | `SelfLearning` | Pseudo-label only confident rejects | MAR; high-confidence labels reliable |
| Heckman control function | `HeckmanClassifier` | Add inverse Mills ratio as a feature | Gaussian selection latent |

`HeckmanClassifier` augments the feature space (not the sample), so it is a standalone classifier — `fit(X_accept, y_accept, X_reject)` then `predict_proba(X)` — rather than a resampler.

## Choosing one

There is no universally best method (see **Benchmark**). Rules of thumb:

- Start with **fuzzy augmentation** — stable and parameter-light.
- Use **parcelling** with `uplift > 1` when policy says rejects are worse than their score implies.
- Use **reweighting** or **Heckman** when you prefer correcting bias to inventing labels.
- Use **extrapolation** when the relationship is local/non-linear.

Always confirm with the benchmark before adopting any of them.
