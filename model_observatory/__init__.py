"""Dependency-light diagnostics for classification systems."""

from .drift import compare_distributions
from .metrics import evaluate
from .schema import PredictionRecord, ValidationError
from .slices import analyze_slices

__all__ = [
    "PredictionRecord",
    "ValidationError",
    "analyze_slices",
    "compare_distributions",
    "evaluate",
]

__version__ = "0.1.0"
