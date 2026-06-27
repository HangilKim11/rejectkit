import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from rejectkit import MaskedRejectBenchmark, plotting
from rejectkit.datasets import make_credit_data


def test_plot_functions_return_axes():
    rng = np.random.default_rng(0)
    ax1 = plotting.plot_score_distributions(rng.random(200), rng.random(200))
    assert hasattr(ax1, "plot")
    ax2 = plotting.plot_ks(rng.integers(0, 2, 200), rng.random(200))
    assert hasattr(ax2, "plot")


def test_plot_benchmark():
    X, y = make_credit_data(n_samples=1200, random_state=0)
    df = MaskedRejectBenchmark(selection="mar", selection_strength=1.5, random_state=0).compare(
        ["fuzzy", "parcelling"], X, y
    )
    ax = plotting.plot_benchmark(df)
    assert hasattr(ax, "bar") or hasattr(ax, "plot")
