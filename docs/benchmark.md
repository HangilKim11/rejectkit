# Benchmark

You can never validate reject inference directly, because rejects have no outcome. `MaskedRejectBenchmark` settles the question **on your own data**: it takes a fully labelled dataset, hides the labels of a synthetically "rejected" subset, and measures how well each method recovers a model close to the *oracle* (trained on the full population) versus the *naive* accepts-only baseline.

```python
from rejectkit import MaskedRejectBenchmark
from rejectkit.datasets import make_credit_data

X, y = make_credit_data(n_samples=4000, random_state=0)
bench = MaskedRejectBenchmark(selection="cutoff", accept_rate=0.6, random_state=0)
print(bench.compare(
    ["simple", "fuzzy", "parcelling", "reweighting",
     "reclassification", "extrapolation", "selflearning", "heckman"],
    X, y,
).round(4))
```

`auc_recovery`: `0` = no better than the naive accepts-only model, `1` = matches the full-data oracle.

## Selection mechanisms

| `selection` | Acceptance depends on | Use it to model |
|---|---|---|
| `"mar"` | observed features only | missing-at-random selection |
| `"mnar"` | features **and** the hidden outcome | the hard case; naive is most biased |
| `"cutoff"` | predicted PD (accept lowest-risk fraction) | a realistic credit policy cutoff |

## How the accept/reject split is simulated

Real labelled data has an outcome for every row, so the harness *creates* rejects:

1. Score each applicant: `acceptance = f(features) (+ outcome term under MNAR) + noise`.
2. Accept the top `accept_rate` fraction; reject the rest.
3. Hide the rejected rows' labels (the *"Masked"* in the name).
4. Train each model on accepts (+ inferred rejects) and score it against the **held-out, fully-labelled** test set.

The rejects are therefore not real declined applicants but labelled rows whose outcome was deliberately hidden — the only way to *measure* recovery, since genuine rejects have no outcome to check against.

## Reading the result honestly

Reject inference is **not** a free lunch. Under pure **MNAR** selection, augmentation methods inherit the accept model's bias and often fail to beat naive — exactly what theory predicts. Under **MAR** and **cutoff** selection, methods such as parcelling and extrapolation can recover a meaningful share of the oracle gap. The point of the harness is to let you discover which regime you are in *before* shipping reject inference on faith.
