# Explainer / 쉬운 설명 / やさしい解説

A beginner-friendly, trilingual (한국어 · English · 日本語) explanation of **what
rejectkit does, what goes in, and what comes out**. For the method catalogue see
[Methods](methods.md); for the benchmark see [Benchmark](benchmark.md).

---

## 1. What problem does it solve? / 어떤 문제를 푸나 / 何を解決するか

**[한국어]** 대출 심사를 떠올려 보자. 신청자를 보고 **승인/거절**을 정한다. 승인한 사람은 나중에 "잘 갚음(good=0) / 연체(bad=1)" 결과를 알게 되지만, **거절한 사람은 대출을 안 줬으니 결과를 영원히 모른다.** 그런데 내년에 부도 예측 모델을 만들려면 정답이 있는 데이터가 필요하고, 그건 **승인자뿐**이다. 승인자는 "괜찮아 보여서" 통과된 치우친 표본이라, 그들만으로 배운 모델은 실제로 문 앞에 오는 **전체 신청자**(위험해 보이는 사람 포함)와 어긋난다. 이를 **표본 선택 편향(sample selection bias)**이라 한다. `rejectkit`는 거절자를 고려해 이 편향을 바로잡는 **reject inference** 기법들을 모은 라이브러리다.

**[English]** Think of loan underwriting. You see an applicant and decide **approve/reject**. For approved applicants you eventually learn the outcome (repaid = good = 0, defaulted = bad = 1); for **rejected applicants you never find out**, because they never got the loan. Yet to build a default-prediction model you need labelled data — and only the **approved** group has labels. Approved applicants are a skewed slice (they already "looked good"), so a model trained on them alone is mismatched against the **whole through-the-door population** it must actually score. This is **sample-selection bias**. `rejectkit` packages the **reject inference** techniques that correct it by accounting for the rejects.

**[日本語]** 与信審査を思い浮かべてほしい。申込者を見て**承認/否認**を決める。承認した人は後で結果（完済 = good = 0、延滞 = bad = 1）が分かるが、**否認した人は融資していないので結果が永遠に分からない**。しかし延滞予測モデルを作るには正解ラベル付きデータが要り、それは**承認者だけ**にある。承認者は「良さそうだから」通った偏った標本なので、それだけで学習したモデルは実際に窓口に来る**全申込者**（リスキーに見える人を含む）とズレる。これが**標本選択バイアス**だ。`rejectkit` は否認者を考慮してこのバイアスを補正する **reject inference** 手法を集めたライブラリ。

---

## 2. Input data / 입력 데이터 / 入力データ

The core input is three pieces: `(X_accept, y_accept, X_reject)`. Rows = applicants, columns = features. Inputs may be pandas, polars, or numpy.

| name | shape | meaning |
|---|---|---|
| `X_accept` | (n_accept, n_features) | features of approved applicants |
| `y_accept` | (n_accept,) | their outcome — `1 = bad`, `0 = good`; **both must be present** |
| `X_reject` | (n_reject, n_features) | features of rejected applicants — **same columns, no label** |

**[한국어]** 핵심 입력은 세 덩어리 `(X_accept, y_accept, X_reject)`다. `y_accept`는 **1=bad(연체), 0=good** 으로 방향이 고정이고 0과 1이 둘 다 있어야 한다. `X_reject`는 `X_accept`와 **열(특징)이 같아야** 하며 정답이 없다. 예측할 신규 신청자 `X_new`도 같은 열 구조다. (벤치마크만은 예외로, 정답이 전부 있는 `(X, y)` 한 덩어리만 받는다.)

**[English]** The core input is the triple `(X_accept, y_accept, X_reject)`. In `y_accept`, the convention is fixed: **1 = bad (default), 0 = good**, and both classes must appear. `X_reject` must have the **same columns** as `X_accept` and carries no label. New applicants `X_new` share the same column layout. (The benchmark is the one exception: it takes a single **fully labelled** `(X, y)`.)

**[日本語]** 中心となる入力は三つ組 `(X_accept, y_accept, X_reject)`。`y_accept` は **1=bad（延滞）, 0=good** と向きが固定で、0 と 1 の両方が必要。`X_reject` は `X_accept` と**同じ列（特徴量）**を持ち、ラベルは無い。予測対象の新規申込者 `X_new` も同じ列構成。（ベンチマークだけは例外で、正解が全部ある `(X, y)` を一つ受け取る。）

---

## 3. Output data / 출력 데이터 / 出力データ

| call | output | meaning |
|---|---|---|
| `clf.predict_proba(X)` | array (n, 2) | column 1 = **P(bad)** |
| `clf.predict(X)` | array (n,) | 0/1 label |
| `method.resample()` | `(X, y, sample_weight)` | augmented through-the-door training set |
| `benchmark.compare(...)` | DataFrame | one row per model, metric columns |
| `feature_drift(...)` | Series | per-feature accept-vs-reject PSI |
| `swap_set(...)` | DataFrame | swap-in/out groups with bad rates |

**[한국어]** 예측은 `predict_proba(X)`가 `(n, 2)` 배열이고 **2열이 부도확률 P(bad)**다. `resample()`은 `(X, y, sample_weight)` 3종을 주는데, 거절자를 채워 넣으며 **행이 늘고 가중치(소수)**가 붙는다(예: fuzzy는 거절자 1명을 good/bad 두 줄로). 벤치마크는 모델별 `auc/ks/gini/auc_recovery` 표를, `feature_drift`는 변수별 PSI를, `swap_set`은 스코어카드 교체 시 누가 들어오고 빠지는지를 돌려준다.

**[English]** For prediction, `predict_proba(X)` is an `(n, 2)` array whose **column 1 is P(bad)**. `resample()` returns the triple `(X, y, sample_weight)`: rejects are folded in, so **rows grow and fractional weights** appear (e.g. fuzzy turns one reject into a good row and a bad row). The benchmark returns a per-model table of `auc/ks/gini/auc_recovery`; `feature_drift` returns per-feature PSI; `swap_set` returns who is swapped in/out when you change scorecards.

**[日本語]** 予測は `predict_proba(X)` が `(n, 2)` 配列で、**2列目が延滞確率 P(bad)**。`resample()` は `(X, y, sample_weight)` の三つ組を返し、否認者を取り込むので**行が増え、小数の重み**が付く（例：fuzzy は否認者1人を good 行と bad 行の2行にする）。ベンチマークはモデル別の `auc/ks/gini/auc_recovery` 表を、`feature_drift` は特徴量別 PSI を、`swap_set` はスコアカード変更時に誰が入れ替わるかを返す。

---

## 4. How to read the benchmark / 벤치마크 해석 / ベンチマークの読み方

`auc_recovery` is the headline number: **0 = no better than the naive accepts-only model, 1 = matches the full-data oracle.** Negative means the method made things worse; `NaN` means the naive model was already at the oracle (nothing to recover).

**[한국어]** `auc_recovery`가 핵심 숫자다. **0이면 naive(승인자만)와 동급, 1이면 oracle(전체 정답)만큼 회복**, 음수면 오히려 악화, `NaN`이면 naive가 이미 oracle급이라 보정할 여지가 없었다는 뜻이다. 즉 "reject inference를 써야 하나, 쓴다면 어떤 방법이 best인가"를 네 데이터로 직접 확인하는 도구다.

**[English]** `auc_recovery` is the key number: **0 = tied with the naive accepts-only model, 1 = recovers the full-data oracle**, negative = the method hurt, `NaN` = naive was already at the oracle (nothing to recover). It lets you decide, on your own data, whether to use reject inference at all and which method is best.

**[日本語]** `auc_recovery` が要となる数値：**0 なら naive（承認者のみ）と同等、1 なら oracle（全データ）まで回復**、負なら悪化、`NaN` は naive が既に oracle 並みで補正の余地が無かったことを示す。つまり「reject inference を使うべきか、使うならどの手法が最良か」を自分のデータで確かめる道具だ。

---

## 5. How are the "rejects" created? / 거절자는 어떻게 만드나 / 否認者はどう作るか

Real labelled data has an outcome for **everyone**, so the benchmark *creates* rejects by simulating a lender's past policy and then **hiding** those applicants' labels:

1. Give every applicant an **acceptance score** = (a mix of their features) + (under MNAR, a bonus if they truly turned out good, a penalty if bad) + random noise.
2. **Accept** the top `accept_rate` fraction; the rest are **rejected**.
3. **Hide** the rejected applicants' labels.
4. Later, score each method against those hidden labels — that hiding is the *"Masked"* in `MaskedRejectBenchmark`.

The `selection` argument picks the policy: `mar` (features only), `mnar` (also depends on the hidden outcome — the realistic, hard case that biases the naive model most), `cutoff` (accept the lowest predicted risk).

**[한국어]** 실데이터는 **모두** 결과가 있으므로, 벤치마크는 과거 심사 정책을 모의해 거절자를 **만들고** 그 라벨을 **가린다**: ① 각 신청자에게 승인 점수 = (특징들의 조합) + (MNAR이면, 실제로 good이면 +·bad이면 −) + 노이즈 를 매긴다. ② 상위 `accept_rate`만 **승인**, 나머지는 **거절**. ③ 거절자의 라벨을 **가린다**. ④ 나중에 가린 라벨로 각 기법을 채점 — 이 "가림"이 `MaskedRejectBenchmark`의 *Masked*다. `selection`: `mar`(특징만), `mnar`(숨은 결과까지 반영, 가장 현실적·가혹), `cutoff`(예측 위험이 낮은 순 승인).

**[日本語]** 実データは**全員**に結果があるため、ベンチマークは過去の審査方針を模擬して否認者を**作り**、そのラベルを**隠す**：① 各申込者に承認スコア =（特徴量の組合せ）+（MNARなら、実際にgoodなら＋・badなら−）+ ノイズ を与える。② 上位 `accept_rate` を**承認**、残りを**否認**。③ 否認者のラベルを**隠す**。④ 後で隠したラベルで各手法を採点 — この「隠す」が `MaskedRejectBenchmark` の *Masked*。`selection`：`mar`（特徴量のみ）、`mnar`（隠れた結果にも依存、最も現実的で厳しい）、`cutoff`（予測リスクの低い順に承認）。

> Note: the "rejects" are **not** real declined applicants — they are labelled applicants whose outcome we deliberately hid. That is the only way to *score* how well a method recovers the truth, since genuine rejects have no outcome to check against.

## 6. Honest caveat / 솔직한 주의 / 正直な注意点

**[한국어]** reject inference는 마법이 아니다. 선택이 결과에 직접 얽힌(MNAR) 어려운 경우엔 추측 기법들이 모델을 **오히려 나쁘게** 만들기도 한다. 그래서 이 라이브러리의 메시지는 "무조건 써라"가 아니라 **"믿고 쓰지 말고 벤치마크로 먼저 확인하라"**다.

**[English]** Reject inference is not magic. When selection is tied to the outcome (MNAR), the guessing methods can make the model **worse**. So the library's message is not "always use it" but **"don't ship it on faith — benchmark first."**

**[日本語]** reject inference は魔法ではない。選択が結果に直結する（MNAR）難しい場合、推測系の手法はモデルを**かえって悪化**させることもある。だからこのライブラリの主張は「常に使え」ではなく、**「信じて使う前に、まずベンチマークで確かめろ」**だ。
