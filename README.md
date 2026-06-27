# rejectkit

**Reject inference for credit scoring — scikit-learn-compatible methods, plus an honest benchmark that tells you whether reject inference actually helps on your data.**

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-beta-orange)

<details>
<summary><b>한국어 요약 (Korean)</b></summary>

<br>

신용 모델은 **승인된 신청자**(나중에 good/bad 결과를 아는 사람)로만 학습하지만, 실제로는 **거절자를 포함한 전체 신청자**를 평가해야 한다 — 이 표본 선택 편향을 바로잡는 기법이 **reject inference**다. `rejectkit`은 이 고전 기법 8가지를 scikit-learn 스타일의 한 API로 묶고, **"그 보정이 내 데이터에서 실제로 도움이 되는지"** 재는 벤치마크(`MaskedRejectBenchmark`)까지 제공한다. 입력은 pandas·polars·numpy 모두 지원. 자세한 한·영·일 설명은 [docs/explainer.md](docs/explainer.md), 실데이터 예제는 [examples/real_data_home_credit.ipynb](examples/real_data_home_credit.ipynb) 참고.

</details>

<details>
<summary><b>日本語要約 (Japanese)</b></summary>

<br>

与信モデルは**承認された申込者**（後で good/bad の結果が分かる人）だけで学習するが、実際には**否認者を含む全申込者**を評価しなければならない — この標本選択バイアスを補正する手法が **reject inference**。`rejectkit` はこの古典的手法8種を scikit-learn 風の単一 API にまとめ、**「その補正が自分のデータで実際に役立つか」**を測るベンチマーク（`MaskedRejectBenchmark`）まで備える。入力は pandas・polars・numpy に対応。詳しい3言語解説は [docs/explainer.md](docs/explainer.md)、実データ例は [examples/real_data_home_credit.ipynb](examples/real_data_home_credit.ipynb) を参照。

</details>

---

## Why this exists

A credit model is trained on **accepted** applicants, whose good/bad outcome you eventually observe. But the model has to score the **whole** through-the-door population — including the applicants you **rejected**, who never get an outcome. Training on accepts only is a textbook case of sample-selection bias. *Reject inference* is the family of techniques that tries to correct it.

These methods are standard in the credit-risk world, yet the Python tooling is missing:

- **R** has [`scoringTools`](https://github.com/adimajo/scoringTools) (`augmentation`, `fuzzy_augmentation`, `parcelling`, `reclassification`, `twins`) — GitHub only.
- **Python** scorecard libraries — `scorecardpy`, `optbinning`, `scorecardbundle` — do WOE/IV binning and logistic scorecards but **skip reject inference entirely**.
- What's left online is one-off research code, not a packaged, tested library.

`rejectkit` fills that gap: eight reject inference methods behind one scikit-learn-style API, a benchmark harness even `scoringTools` lacks, plus drift diagnostics and plotting.

## Install

```bash
pip install -e .              # core: numpy, pandas, scikit-learn
pip install -e ".[plot]"      # + matplotlib plotting helpers
pip install -e ".[polars]"    # + polars input support
```

## Quickstart

```python
from sklearn.linear_model import LogisticRegression
from rejectkit import RejectInferenceClassifier

# X_accept, y_accept: accepted applicants and their good(0)/bad(1) outcomes
# X_reject:           rejected applicants — features only, no labels
clf = RejectInferenceClassifier(
    estimator=LogisticRegression(max_iter=1000),
    method="parcelling",
    method_params={"uplift": 1.3},   # assume rejects are ~30% worse per score band
)
clf.fit(X_accept, y_accept, X_reject)
pd_bad = clf.predict_proba(X_new)[:, 1]
```

Just want the augmented training sample for your own pipeline?

```python
from rejectkit import FuzzyAugmentation

X_aug, y_aug, sample_weight = (
    FuzzyAugmentation(LogisticRegression(max_iter=1000))
    .fit_resample(X_accept, y_accept, X_reject)
)
```

Inputs may be **pandas, polars, or numpy**.

## Methods

| Method | Class | Core idea | Assumption |
|---|---|---|---|
| Simple augmentation | `SimpleAugmentation` | Hard 0/1 label by score cutoff | Accept model ranks rejects |
| Fuzzy augmentation | `FuzzyAugmentation` | Two weighted rows per reject (P(bad), P(good)) | MAR; smooth labels |
| Parcelling | `Parcelling` | Per-score-band bad rate × `uplift` | Rejects worse by a fixed factor |
| Reclassification | `Reclassification` | Iteratively relabel & refit | Labels converge |
| Extrapolation / twins | `Extrapolation` | Local bad rate of nearest accepts | Similar applicants behave alike |
| Inverse-propensity reweighting | `Reweighting` | Reweight accepts by `1/P(accept)` | MAR; invents no labels |
| Self-training | `SelfLearning` | Pseudo-label only confident rejects | MAR; confident labels reliable |
| Heckman control function | `HeckmanClassifier` | Add inverse Mills ratio as a feature | Gaussian selection latent |

All resamplers share `fit(X_accept, y_accept, X_reject)` → `resample()` returning `(X, y, sample_weight)`. `HeckmanClassifier` augments the feature space, so it is a standalone classifier rather than a resampler.

## Does reject inference actually help? Measure it.

You can never validate reject inference directly, because rejects have no outcome — the literature is genuinely split on whether it helps at all. `MaskedRejectBenchmark` settles the question **on your own data**: it hides the labels of a synthetically "rejected" subset of a labelled dataset and checks how well each method recovers a model close to the *oracle* (trained on the full population) versus the *naive* accepts-only baseline.

```python
from rejectkit import MaskedRejectBenchmark
from rejectkit.datasets import make_credit_data

X, y = make_credit_data(n_samples=4000, random_state=0)
bench = MaskedRejectBenchmark(selection="mnar", accept_rate=0.6, random_state=0)
print(bench.compare(
    ["fuzzy", "parcelling", "reweighting", "extrapolation", "selflearning", "heckman"],
    X, y,
).round(4))
```

```
                     auc      ks    gini  auc_recovery
oracle            0.8203  0.4911  0.6406        1.0000
naive             0.7488  0.3651  0.4975        0.0000
fuzzy             0.7488  0.3663  0.4977        0.0010
parcelling        0.7404  0.3468  0.4809       -0.1161
reweighting       0.7249  0.3290  0.4498       -0.3334
extrapolation     0.6989  0.2889  0.3977       -0.6973
selflearning      0.7124  0.3093  0.4248       -0.5080
heckman           0.7457  0.3559  0.4914       -0.0424
```

`auc_recovery`: `0` = no better than the naive accepts-only model, `1` = matches the full-data oracle.

Read this honestly. Selection here is **MNAR** (acceptance depends on the hidden outcome), so naive is badly biased (0.749 vs the 0.820 oracle) — yet the augmentation methods barely move it and several *hurt*; only Heckman nearly holds the naive line. That is what theory predicts when selection depends on the outcome: **reject inference is not a free lunch.** Switch to `selection="mar"` or `selection="cutoff"` and the verdict often flips the other way — frequently the naive model is *already* at the oracle, so `auc_recovery` returns `NaN` (no gap to recover) and reject inference is simply unnecessary. The harness exists so you find out *before* you ship it.

Selection mechanisms: `"mar"` (features only), `"mnar"` (features + hidden outcome), `"cutoff"` (accept the lowest-PD fraction — a realistic credit policy).

## Diagnostics & plotting

```python
from rejectkit.diagnostics import feature_drift, swap_set, psi
feature_drift(X_accept, X_reject)        # per-feature accept-vs-reject PSI, worst first
swap_set(y, score_old, score_new, c_old, c_new)   # who a new scorecard swaps in/out

from rejectkit import plotting            # needs [plot]
plotting.plot_benchmark(results)
plotting.plot_score_distributions(score_accept, score_reject)
plotting.plot_ks(y_true, y_score)
```

## Caveats

- Augmentation methods infer reject labels from a model fitted on the (biased) accepts, so they cannot escape strong **MNAR** selection on their own.
- Reject inference often affects **calibration** more than **ranking** (AUC). Evaluate the metric you care about.
- Always benchmark before adopting. `rejectkit` makes that one function call.

## Documentation

Build the docs site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Examples

- `examples/quickstart.py` — 60-second tour (single model + benchmark).
- `examples/walkthrough.ipynb` — every function on sample data (trilingual KO/EN/JA, executed).
- `examples/real_data_home_credit.ipynb` — **applied to the real Kaggle Home Credit dataset**: under MNAR selection the naive model collapses (AUC 0.74 → 0.57) and reject inference recovers ~7–8% of the gap; under MAR/cutoff it is unnecessary (trilingual, executed).

## Roadmap

- **v0.1** — core augmentation/parcelling/reweighting, `RejectInferenceClassifier`, benchmark. ✅
- **v0.2** — reclassification, extrapolation / twins. ✅
- **v0.3** — self-training, Heckman, polars, plotting, drift diagnostics, docs. ✅
- **Next** — calibration-focused benchmark metrics, deep generative reject inference (optional extra), PyPI release.

## References

- Hand & Henley (1993), *Can reject inference ever work?*
- Crook & Banasik (2004), *Does reject inference really improve the performance of application scoring models?*
- Lopes, *Should we "reject" Reject Inference? An empirical study.*
- `scoringTools` (R): https://github.com/adimajo/scoringTools

## License

MIT — see [LICENSE](LICENSE).
