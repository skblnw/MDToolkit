"""Integrated correlation analysis for biomolecular dynamics."""

import numpy as np
import logging
from typing import Optional, Union, Dict, Tuple
import MDAnalysis as mda
from MDAnalysis.analysis import align
from scipy import signal
from pathlib import Path

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator
from .covariance import calculate_covariance_matrix

logger = logging.getLogger(__name__)


class CorrelationAnalysis:
    """
    Comprehensive correlation analysis for MD trajectories.
    
    Integrates cross-correlation, auto-correlation, and
    generalized correlation coefficient calculations.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection: str = "protein and name CA",
        align: bool = True,
        align_selection: Optional[str] = None
    ):
        """
        Initialize correlation analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection : str, default "protein and name CA"
            Selection for correlation analysis
        align : bool, default True
            Whether to align trajectory first
        align_selection : str, optional
            Selection for alignment. If None, uses selection
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
            self.handler = trajectory
        else:
            self.universe = trajectory
            self.handler = None
            
        self.selection = selection
        self.atoms = self.universe.select_atoms(selection)
        self.n_atoms = len(self.atoms)
        
        # Alignment
        if align:
            align_sel = align_selection or selection
            self._align_trajectory(align_sel)
            
        self.positions = None
        self.correlation_matrix = None
        self.covariance_matrix = None
        
    def _align_trajectory(self, selection: str) -> None:
        """Align trajectory to first frame."""
        ref = self.universe.copy()
        ref.trajectory[0]
        
        aligner = align.AlignTraj(
            self.universe,
            ref,
            select=selection,
            in_memory=False
        ).run()
        
        logger.info(f"Aligned trajectory using selection: {selection}")
    
    @timing_decorator
    def extract_positions(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> np.ndarray:
        """
        Extract positions for correlation analysis.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int, default 1
            Step between frames
            
        Returns
        -------
        np.ndarray
            Positions array (n_frames, n_atoms, 3)
        """
        positions = []
        
        for ts in self.universe.trajectory[start:stop:step]:
            positions.append(self.atoms.positions.copy())
            
        self.positions = np.array(positions)
        
        # Center positions
        mean_pos = np.mean(self.positions, axis=0)
        self.positions -= mean_pos
        
        logger.info(f"Extracted positions: shape {self.positions.shape}")
        
        return self.positions
    
    @timing_decorator
    def calculate_correlation_matrix(
        self,
        method: str = "pearson",
        lag: int = 0
    ) -> np.ndarray:
        """
        Calculate correlation matrix between atoms.
        
        Parameters
        ----------
        method : str, default "pearson"
            Correlation method ("pearson", "mi", "gcc")
        lag : int, default 0
            Time lag for correlation
            
        Returns
        -------
        np.ndarray
            Correlation matrix (n_atoms, n_atoms)
        """
        if self.positions is None:
            self.extract_positions()
            
        n_frames, n_atoms, _ = self.positions.shape
        
        if method == "pearson":
            # Pearson correlation
            corr_matrix = np.zeros((n_atoms, n_atoms))
            
            for i in range(n_atoms):
                for j in range(i, n_atoms):
                    # Calculate correlation for each coordinate
                    corr = 0
                    for dim in range(3):
                        if lag == 0:
                            c = np.corrcoef(
                                self.positions[:, i, dim],
                                self.positions[:, j, dim]
                            )[0, 1]
                        else:
                            # Lagged correlation
                            x = self.positions[:-lag, i, dim]
                            y = self.positions[lag:, j, dim]
                            c = np.corrcoef(x, y)[0, 1]
                        
                        corr += c
                    
                    corr_matrix[i, j] = corr / 3
                    corr_matrix[j, i] = corr_matrix[i, j]
                    
        elif method == "mi":
            # Mutual information
            corr_matrix = self._calculate_mutual_information()
            
        elif method == "gcc":
            # Generalized correlation coefficient
            corr_matrix = self._calculate_gcc()
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.correlation_matrix = corr_matrix
        
        return corr_matrix
    
    def _calculate_mutual_information(self) -> np.ndarray:
        """Calculate mutual information between atoms."""
        from sklearn.metrics import mutual_info_score
        
        n_atoms = self.n_atoms
        mi_matrix = np.zeros((n_atoms, n_atoms))
        
        # Discretize positions for MI calculation
        n_bins = 20
        
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                mi = 0
                for dim in range(3):
                    # Discretize positions
                    x = np.digitize(
                        self.positions[:, i, dim],
                        bins=np.linspace(
                            self.positions[:, i, dim].min(),
                            self.positions[:, i, dim].max(),
                            n_bins
                        )
                    )
                    y = np.digitize(
                        self.positions[:, j, dim],
                        bins=np.linspace(
                            self.positions[:, j, dim].min(),
                            self.positions[:, j, dim].max(),
                            n_bins
                        )
                    )
                    
                    mi += mutual_info_score(x, y)
                
                mi_matrix[i, j] = mi / 3
                mi_matrix[j, i] = mi_matrix[i, j]
        
        # Normalize
        mi_matrix = mi_matrix / np.max(mi_matrix)
        
        return mi_matrix
    
    def _calculate_gcc(self) -> np.ndarray:
        """Calculate generalized correlation coefficient."""
        n_atoms = self.n_atoms
        gcc_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                # Calculate GCC using distance correlation
                gcc = self._distance_correlation(
                    self.positions[:, i, :],
                    self.positions[:, j, :]
                )
                
                gcc_matrix[i, j] = gcc
                gcc_matrix[j, i] = gcc
        
        return gcc_matrix
    
    @staticmethod
    def _distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate distance correlation between two variables.
        
        Parameters
        ----------
        X : np.ndarray
            First variable (n_samples, n_features)
        Y : np.ndarray
            Second variable (n_samples, n_features)
            
        Returns
        -------
        float
            Distance correlation
        """
        n = len(X)
        
        # Compute distance matrices
        a = np.sqrt(np.sum((X[:, None] - X[None, :])**2, axis=2))
        b = np.sqrt(np.sum((Y[:, None] - Y[None, :])**2, axis=2))
        
        # Double center the distance matrices
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        
        # Calculate distance covariance and variances
        dcov2 = np.sum(A * B) / (n * n)
        dvar2_X = np.sum(A * A) / (n * n)
        dvar2_Y = np.sum(B * B) / (n * n)
        
        # Distance correlation
        if dvar2_X > 0 and dvar2_Y > 0:
            dcor = np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar2_X) * np.sqrt(dvar2_Y))
        else:
            dcor = 0
            
        return dcor
    
    def calculate_dynamic_correlation(
        self,
        window_size: int = 100,
        stride: int = 10
    ) -> np.ndarray:
        """
        Calculate time-resolved correlation.
        
        Parameters
        ----------
        window_size : int, default 100
            Window size in frames
        stride : int, default 10
            Stride between windows
            
        Returns
        -------
        np.ndarray
            Dynamic correlation (n_windows, n_atoms, n_atoms)
        """
        if self.positions is None:
            self.extract_positions()
            
        n_frames = len(self.positions)
        n_windows = (n_frames - window_size) // stride + 1
        
        dynamic_corr = np.zeros((n_windows, self.n_atoms, self.n_atoms))
        
        for w in range(n_windows):
            start = w * stride
            end = start + window_size
            
            # Extract window positions
            window_pos = self.positions[start:end]
            
            # Calculate correlation for this window
            for i in range(self.n_atoms):
                for j in range(i, self.n_atoms):
                    corr = 0
                    for dim in range(3):
                        c = np.corrcoef(
                            window_pos[:, i, dim],
                            window_pos[:, j, dim]
                        )[0, 1]
                        corr += c
                    
                    dynamic_corr[w, i, j] = corr / 3
                    dynamic_corr[w, j, i] = dynamic_corr[w, i, j]
        
        return dynamic_corr
    
    def calculate_residue_correlation(self) -> np.ndarray:
        """
        Calculate residue-level correlation.
        
        Returns
        -------
        np.ndarray
            Residue correlation matrix
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
            
        # Get residue information
        residues = self.atoms.residues
        n_residues = len(residues)
        
        # Initialize residue correlation matrix
        res_corr = np.zeros((n_residues, n_residues))
        
        # Average over atoms in each residue
        for i, res_i in enumerate(residues):
            atoms_i = res_i.atoms.intersection(self.atoms)
            idx_i = [self.atoms.indices.tolist().index(a.index) 
                    for a in atoms_i]
            
            for j, res_j in enumerate(residues):
                atoms_j = res_j.atoms.intersection(self.atoms)
                idx_j = [self.atoms.indices.tolist().index(a.index)
                        for a in atoms_j]
                
                # Average correlation between residues
                if idx_i and idx_j:
                    corr_block = self.correlation_matrix[np.ix_(idx_i, idx_j)]
                    res_corr[i, j] = np.mean(corr_block)
        
        return res_corr
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        save_positions: bool = False
    ) -> None:
        """
        Save correlation analysis results.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        save_positions : bool, default False
            Whether to save extracted positions
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.correlation_matrix is not None:
            np.save(
                output_dir / "correlation_matrix.npy",
                self.correlation_matrix
            )
            
        if self.covariance_matrix is not None:
            np.save(
                output_dir / "covariance_matrix.npy",
                self.covariance_matrix
            )
            
        if save_positions and self.positions is not None:
            np.save(
                output_dir / "positions.npy",
                self.positions
            )
            
        logger.info(f"Saved correlation results to {output_dir}")


def calculate_correlation_matrix(
    universe: mda.Universe,
    selection: str = "protein and name CA",
    method: str = "pearson"
) -> np.ndarray:
    """
    Quick correlation matrix calculation.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str, default "protein and name CA"
        Atom selection
    method : str, default "pearson"
        Correlation method
        
    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    ca = CorrelationAnalysis(universe, selection)
    ca.extract_positions()
    return ca.calculate_correlation_matrix(method=method)


def calculate_cross_correlation(
    positions1: np.ndarray,
    positions2: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate cross-correlation between two position arrays.
    
    Parameters
    ----------
    positions1 : np.ndarray
        First position array (n_frames, n_atoms, 3)
    positions2 : np.ndarray
        Second position array (n_frames, n_atoms, 3)
    normalize : bool, default True
        Whether to normalize correlation
        
    Returns
    -------
    np.ndarray
        Cross-correlation matrix
    """
    n_atoms1 = positions1.shape[1]
    n_atoms2 = positions2.shape[1]
    
    cross_corr = np.zeros((n_atoms1, n_atoms2))
    
    for i in range(n_atoms1):
        for j in range(n_atoms2):
            corr = 0
            for dim in range(3):
                c = np.corrcoef(
                    positions1[:, i, dim],
                    positions2[:, j, dim]
                )[0, 1]
                corr += c
            
            cross_corr[i, j] = corr / 3
    
    if normalize:
        cross_corr = cross_corr / np.max(np.abs(cross_corr))
    
    return cross_corr