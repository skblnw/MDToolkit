"""Visualization module for MD analysis results."""

from .plots import (
    plot_rmsd,
    plot_rmsf,
    plot_contacts,
    plot_correlation_matrix,
    plot_pca,
    plot_free_energy_landscape,
    plot_timeseries
)
from .plot_templates import PlotTemplate, PublicationPlot
from .colors import get_colormap, get_color_palette

__all__ = [
    "plot_rmsd",
    "plot_rmsf",
    "plot_contacts",
    "plot_correlation_matrix",
    "plot_pca",
    "plot_free_energy_landscape",
    "plot_timeseries",
    "PlotTemplate",
    "PublicationPlot",
    "get_colormap",
    "get_color_palette",
]