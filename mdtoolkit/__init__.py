"""
MD-Toolkit: A comprehensive molecular dynamics analysis toolkit

This package provides a unified interface for analyzing molecular dynamics
trajectories with emphasis on structural dynamics, correlation analysis,
and biomolecular systems.
"""

__version__ = "1.0.0"
__author__ = "MD-Toolkit Contributors"

from . import core
from . import structure
from . import dynamics
from . import visualization
from . import workflows

__all__ = [
    "core",
    "structure", 
    "dynamics",
    "visualization",
    "workflows",
]