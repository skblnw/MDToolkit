"""Covariance analysis for MD trajectories."""

import numpy as np
import logging
from typing import Optional, Union, Dict
import MDAnalysis as mda
from scipy import linalg

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class CovarianceAnalysis:
    """
    Covariance analysis for molecular dynamics trajectories.
    
    Calculates covariance matrices at different levels of granularity.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection: str = "protein and name CA",
        align: bool = True
    ):
        """
        Initialize covariance analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection : str
            Atom selection
        align : bool
            Whether to align trajectory first
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection = selection
        self.atoms = self.universe.select_atoms(selection)
        self.n_atoms = len(self.atoms)
        
        if align:
            self._align_trajectory()
            
        self.positions = None
        self.covariance = None
        
    def _align_trajectory(self) -> None:
        """Align trajectory to first frame."""
        from MDAnalysis.analysis import align
        
        ref = self.universe.copy()
        ref.trajectory[0]
        
        aligner = align.AlignTraj(
            self.universe,
            ref,
            select=self.selection,
            in_memory=False
        ).run()
        
    @timing_decorator
    def calculate_covariance(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> np.ndarray:
        """
        Calculate covariance matrix.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int
            Step between frames
            
        Returns
        -------
        np.ndarray
            Covariance matrix (3N x 3N)
        """
        # Extract positions
        positions = []
        for ts in self.universe.trajectory[start:stop:step]:
            positions.append(self.atoms.positions.flatten())
        
        positions = np.array(positions)
        
        # Center positions
        mean_pos = np.mean(positions, axis=0)
        centered = positions - mean_pos
        
        # Calculate covariance
        n_frames = len(centered)
        self.covariance = np.dot(centered.T, centered) / (n_frames - 1)
        
        self.positions = centered
        
        return self.covariance
    
    def calculate_distance_covariance(self) -> np.ndarray:
        """
        Calculate distance covariance matrix.
        
        Returns
        -------
        np.ndarray
            Distance covariance matrix (N x N)
        """
        if self.positions is None:
            self.calculate_covariance()
            
        n_frames = len(self.positions)
        positions_3d = self.positions.reshape(n_frames, self.n_atoms, 3)
        
        # Calculate pairwise distances for each frame
        dist_matrix = np.zeros((self.n_atoms, self.n_atoms))
        
        for frame in positions_3d:
            for i in range(self.n_atoms):
                for j in range(i+1, self.n_atoms):
                    dist = np.linalg.norm(frame[i] - frame[j])
                    dist_matrix[i, j] += dist
                    dist_matrix[j, i] = dist_matrix[i, j]
        
        dist_matrix /= n_frames
        
        return dist_matrix
    
    def get_eigenvalues(self) -> tuple:
        """
        Get eigenvalues and eigenvectors of covariance matrix.
        
        Returns
        -------
        tuple
            (eigenvalues, eigenvectors)
        """
        if self.covariance is None:
            self.calculate_covariance()
            
        eigenvalues, eigenvectors = linalg.eigh(self.covariance)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors


def calculate_covariance_matrix(
    universe: mda.Universe,
    selection: str = "protein and name CA"
) -> np.ndarray:
    """
    Quick covariance matrix calculation.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str
        Atom selection
        
    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    cov_analysis = CovarianceAnalysis(universe, selection)
    return cov_analysis.calculate_covariance()


def calculate_residue_covariance(
    universe: mda.Universe,
    selection: str = "protein and name CA"
) -> np.ndarray:
    """
    Calculate residue-level covariance.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str
        Atom selection
        
    Returns
    -------
    np.ndarray
        Residue covariance matrix
    """
    atoms = universe.select_atoms(selection)
    residues = atoms.residues
    n_residues = len(residues)
    
    # Get positions for each residue
    residue_positions = []
    
    for ts in universe.trajectory:
        res_pos = []
        for res in residues:
            res_atoms = res.atoms.intersection(atoms)
            if len(res_atoms) > 0:
                res_pos.append(res_atoms.center_of_mass())
            else:
                res_pos.append(np.zeros(3))
        residue_positions.append(res_pos)
    
    residue_positions = np.array(residue_positions)
    
    # Flatten and center
    n_frames = len(residue_positions)
    flat_positions = residue_positions.reshape(n_frames, -1)
    mean_pos = np.mean(flat_positions, axis=0)
    centered = flat_positions - mean_pos
    
    # Calculate covariance
    covariance = np.dot(centered.T, centered) / (n_frames - 1)
    
    # Reshape to residue blocks
    res_cov = covariance.reshape(n_residues, 3, n_residues, 3)
    
    # Calculate magnitude of covariance between residues
    res_cov_mag = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        for j in range(n_residues):
            res_cov_mag[i, j] = np.trace(res_cov[i, :, j, :])
    
    return res_cov_mag