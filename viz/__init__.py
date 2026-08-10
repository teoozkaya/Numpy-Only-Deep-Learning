"""Plotting helpers: loss curves, weight statistics, decision boundaries."""

from viz.boundary import plot_decision_boundary
from viz.curves import plot_history, plot_loss_curves
from viz.weights import plot_filters, plot_weight_histogram

__all__ = [
    "plot_decision_boundary",
    "plot_history",
    "plot_loss_curves",
    "plot_filters",
    "plot_weight_histogram",
]
