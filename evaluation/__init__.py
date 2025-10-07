"""Evaluation Module for Model Performance Assessment"""

from .metrics import evaluate_model, calculate_metrics, plot_confusion_matrix, plot_roc_curve
from .stability import ExplanationStabilityTracker

__all__ = [
    "evaluate_model",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "ExplanationStabilityTracker"
]
