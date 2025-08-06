"""Dynamics and correlation analysis module for MD trajectories."""

from .correlation import (
    CorrelationAnalysis,
    calculate_correlation_matrix,
    calculate_cross_correlation
)
from .pca import (
    PCAAnalysis,
    perform_pca,
    validate_pca_with_sklearn
)
from .covariance import (
    CovarianceAnalysis,
    calculate_covariance_matrix,
    calculate_residue_covariance
)

__all__ = [
    "CorrelationAnalysis",
    "calculate_correlation_matrix",
    "calculate_cross_correlation",
    "PCAAnalysis",
    "perform_pca",
    "validate_pca_with_sklearn",
    "CovarianceAnalysis",
    "calculate_covariance_matrix",
    "calculate_residue_covariance",
]