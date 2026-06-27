"""rejectkit — reject inference for credit scoring, scikit-learn style.

Reject inference corrects the sampling bias that arises when a credit model is
trained only on *accepted* applicants whose good/bad outcome is observed, while
the *rejected* applicants — who are also part of the through-the-door
population the model must score — are silently dropped.

``rejectkit`` provides the classic reject inference methods behind a single,
scikit-learn-compatible API, plus a benchmark harness that lets you measure —
on your own data — whether reject inference actually helps.
"""

from . import datasets, diagnostics, plotting
from .base import BaseRejectInferencer
from .benchmark import MaskedRejectBenchmark
from .estimator import RejectInferenceClassifier, get_inferencer
from .methods import (
    Extrapolation,
    FuzzyAugmentation,
    HeckmanClassifier,
    Parcelling,
    Reclassification,
    Reweighting,
    SelfLearning,
    SimpleAugmentation,
)

__version__ = "0.3.0"

__all__ = [
    "BaseRejectInferencer",
    "SimpleAugmentation",
    "FuzzyAugmentation",
    "Parcelling",
    "Reweighting",
    "Reclassification",
    "Extrapolation",
    "SelfLearning",
    "HeckmanClassifier",
    "RejectInferenceClassifier",
    "get_inferencer",
    "MaskedRejectBenchmark",
    "diagnostics",
    "datasets",
    "plotting",
    "__version__",
]
