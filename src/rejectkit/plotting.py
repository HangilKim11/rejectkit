"""Optional matplotlib plotting helpers.

Importing this module is cheap; matplotlib is only imported when a plotting
function is actually called. Install the extra with ``pip install rejectkit[plot]``.
"""

from __future__ import annotations

import numpy as np

from ._compat import to_numpy_1d


def _get_ax(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install rejectkit[plot]`."
        ) from exc
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_benchmark(results, metric: str = "auc_recovery", ax=None):
    """Bar chart of a :class:`MaskedRejectBenchmark` results table."""
    ax = _get_ax(ax)
    df = results
    if metric == "auc_recovery" and "oracle" in df.index:
        df = df.drop(index="oracle")
    vals = df[metric]
    colors = ["#999999" if i == "naive" else "#3366cc" for i in vals.index]
    ax.bar(range(len(vals)), vals.to_numpy(), color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(metric)
    ax.set_title("Reject inference benchmark")
    return ax


def plot_score_distributions(score_accept, score_reject, bins: int = 30, ax=None):
    """Overlaid histograms of accept vs reject scores."""
    ax = _get_ax(ax)
    ax.hist(to_numpy_1d(score_accept), bins=bins, alpha=0.5, density=True, label="accept")
    ax.hist(to_numpy_1d(score_reject), bins=bins, alpha=0.5, density=True, label="reject")
    ax.set_xlabel("score / P(bad)")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Accept vs reject score distribution")
    return ax


def plot_ks(y_true, y_score, ax=None):
    """Plot the KS curve (cumulative bad vs good across the score threshold)."""
    from sklearn.metrics import roc_curve

    ax = _get_ax(ax)
    y_true = to_numpy_1d(y_true)
    y_score = to_numpy_1d(y_score).astype(float)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    finite = np.isfinite(thr)
    ks = float(np.max(tpr - fpr))
    k = int(np.argmax(tpr - fpr))
    ax.plot(thr[finite], tpr[finite], label="cumulative bad (TPR)")
    ax.plot(thr[finite], fpr[finite], label="cumulative good (FPR)")
    if np.isfinite(thr[k]):
        ax.axvline(thr[k], color="k", ls="--", lw=0.8, label=f"KS = {ks:.3f}")
    ax.set_xlabel("threshold on P(bad)")
    ax.set_ylabel("cumulative rate")
    ax.invert_xaxis()
    ax.legend()
    ax.set_title("KS curve")
    return ax
