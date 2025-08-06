"""Principal Component Analysis for MD trajectories with sklearn validation."""

import numpy as np
import logging
from typing import Optional, Union, Dict, Tuple, List
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
from sklearn.decomposition import PCA as sklearn_PCA
from pathlib import Path

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class PCAAnalysis:
    """
    Principal Component Analysis for MD trajectories.
    
    Integrates MDAnalysis PCA with sklearn validation capabilities
    for comprehensive dimensionality reduction analysis.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection: str = "protein and name CA",
        align: bool = True,
        align_selection: Optional[str] = None
    ):
        """
        Initialize PCA analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection : str, default "protein and name CA"
            Selection for PCA
        align : bool, default True
            Whether to align trajectory first
        align_selection : str, optional
            Selection for alignment. If None, uses selection
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection = selection
        self.atoms = self.universe.select_atoms(selection)
        self.n_atoms = len(self.atoms)
        
        # Alignment
        if align:
            align_sel = align_selection or selection
            self._align_trajectory(align_sel)
            
        self.pca_mda = None
        self.pca_sklearn = None
        self.transformed = None
        self.eigenvectors = None
        self.eigenvalues = None
        
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
    def run_mda_pca(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        mean: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run PCA using MDAnalysis.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int, default 1
            Step between frames
        mean : np.ndarray, optional
            Mean structure for centering
            
        Returns
        -------
        dict
            PCA results from MDAnalysis
        """
        # Run MDAnalysis PCA
        self.pca_mda = pca.PCA(
            self.universe,
            select=self.selection,
            align=False,  # Already aligned
            mean=mean
        )
        
        self.pca_mda.run(start=start, stop=stop, step=step)
        
        # Store results
        results = {
            'cumulated_variance': self.pca_mda.cumulated_variance,
            'variance': self.pca_mda.variance,
            'p_components': self.pca_mda.p_components,
            'n_components': len(self.pca_mda.p_components),
            'transformed': self.pca_mda.transformed
        }
        
        self.eigenvectors = self.pca_mda.p_components
        self.eigenvalues = self.pca_mda.variance
        self.transformed = self.pca_mda.transformed
        
        logger.info(f"MDAnalysis PCA completed with {results['n_components']} components")
        
        return results
    
    @timing_decorator
    def run_sklearn_pca(
        self,
        n_components: Optional[int] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> Dict:
        """
        Run PCA using sklearn for validation.
        
        Parameters
        ----------
        n_components : int, optional
            Number of components. If None, keeps all
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int, default 1
            Step between frames
            
        Returns
        -------
        dict
            PCA results from sklearn
        """
        # Extract positions
        positions = []
        for ts in self.universe.trajectory[start:stop:step]:
            positions.append(self.atoms.positions.flatten())
        
        X = np.array(positions)
        
        # Run sklearn PCA
        self.pca_sklearn = sklearn_PCA(n_components=n_components)
        self.transformed_sklearn = self.pca_sklearn.fit_transform(X)
        
        # Store results
        results = {
            'explained_variance': self.pca_sklearn.explained_variance_,
            'explained_variance_ratio': self.pca_sklearn.explained_variance_ratio_,
            'cumsum_variance': np.cumsum(self.pca_sklearn.explained_variance_ratio_),
            'components': self.pca_sklearn.components_,
            'n_components': self.pca_sklearn.n_components_,
            'transformed': self.transformed_sklearn
        }
        
        logger.info(f"sklearn PCA completed with {results['n_components']} components")
        
        return results
    
    def validate_pca(self, tolerance: float = 1e-3) -> Dict:
        """
        Validate PCA results between MDAnalysis and sklearn.
        
        Parameters
        ----------
        tolerance : float, default 1e-3
            Tolerance for comparison
            
        Returns
        -------
        dict
            Validation results
        """
        if self.pca_mda is None or self.pca_sklearn is None:
            raise RuntimeError("Must run both MDAnalysis and sklearn PCA first")
        
        validation = {}
        
        # Compare variance explained
        n_comp = min(len(self.pca_mda.variance), 
                    len(self.pca_sklearn.explained_variance_))
        
        mda_var = self.pca_mda.variance[:n_comp]
        sklearn_var = self.pca_sklearn.explained_variance_[:n_comp]
        
        # Normalize for comparison
        mda_var_norm = mda_var / np.sum(mda_var)
        sklearn_var_norm = sklearn_var / np.sum(sklearn_var)
        
        var_diff = np.abs(mda_var_norm - sklearn_var_norm)
        
        validation['variance_match'] = np.all(var_diff < tolerance)
        validation['max_variance_diff'] = np.max(var_diff)
        validation['mean_variance_diff'] = np.mean(var_diff)
        
        # Compare projections (accounting for sign flip)
        n_samples = min(self.transformed.shape[0], 
                       self.transformed_sklearn.shape[0])
        
        projection_corr = []
        for i in range(min(3, n_comp)):  # Check first 3 PCs
            mda_pc = self.transformed[:n_samples, i]
            sklearn_pc = self.transformed_sklearn[:n_samples, i]
            
            # Check correlation (absolute value for sign flip)
            corr = np.abs(np.corrcoef(mda_pc, sklearn_pc)[0, 1])
            projection_corr.append(corr)
        
        validation['projection_correlations'] = projection_corr
        validation['projection_match'] = np.all(np.array(projection_corr) > 0.99)
        
        logger.info(f"PCA validation: variance_match={validation['variance_match']}, "
                   f"projection_match={validation['projection_match']}")
        
        return validation
    
    def calculate_cosine_content(
        self,
        n_components: int = 3
    ) -> np.ndarray:
        """
        Calculate cosine content to check for random diffusion.
        
        Parameters
        ----------
        n_components : int, default 3
            Number of components to check
            
        Returns
        -------
        np.ndarray
            Cosine content for each component
        """
        if self.transformed is None:
            raise RuntimeError("Must run PCA first")
        
        cosine_content = np.zeros(n_components)
        
        for i in range(n_components):
            pc = self.transformed[:, i]
            n_frames = len(pc)
            
            # Calculate cosine content
            t = np.arange(n_frames)
            cos_arg = np.pi * t / (n_frames - 1)
            
            numerator = np.sum(pc * np.cos(cos_arg))**2
            denominator = n_frames * np.sum(pc**2)
            
            cosine_content[i] = numerator / denominator
        
        return cosine_content
    
    def project_trajectory(
        self,
        other_universe: mda.Universe,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Project another trajectory onto PCA space.
        
        Parameters
        ----------
        other_universe : MDAnalysis.Universe
            Trajectory to project
        n_components : int, optional
            Number of components to project onto
            
        Returns
        -------
        np.ndarray
            Projected coordinates
        """
        if self.pca_mda is None:
            raise RuntimeError("Must run MDAnalysis PCA first")
        
        # Select atoms
        atoms = other_universe.select_atoms(self.selection)
        
        # Project
        projected = []
        for ts in other_universe.trajectory:
            # Center positions
            pos = atoms.positions - self.pca_mda.mean
            pos_flat = pos.flatten()
            
            # Project onto PCs
            if n_components is None:
                proj = np.dot(pos_flat, self.pca_mda.p_components.T)
            else:
                proj = np.dot(pos_flat, self.pca_mda.p_components[:n_components].T)
            
            projected.append(proj)
        
        return np.array(projected)
    
    def get_extreme_structures(
        self,
        component: int = 0,
        n_structures: int = 5
    ) -> Dict[str, List[int]]:
        """
        Get frames with extreme PC values.
        
        Parameters
        ----------
        component : int, default 0
            PC component to analyze
        n_structures : int, default 5
            Number of extreme structures
            
        Returns
        -------
        dict
            Frame indices for minimum and maximum PC values
        """
        if self.transformed is None:
            raise RuntimeError("Must run PCA first")
        
        pc_values = self.transformed[:, component]
        
        # Get extreme frames
        min_indices = np.argsort(pc_values)[:n_structures]
        max_indices = np.argsort(pc_values)[-n_structures:]
        
        return {
            'min_frames': min_indices.tolist(),
            'max_frames': max_indices.tolist(),
            'min_values': pc_values[min_indices].tolist(),
            'max_values': pc_values[max_indices].tolist()
        }
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        save_projections: bool = True
    ) -> None:
        """
        Save PCA results.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        save_projections : bool, default True
            Whether to save projected coordinates
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save eigenvectors and eigenvalues
        if self.eigenvectors is not None:
            np.save(output_dir / "eigenvectors.npy", self.eigenvectors)
            
        if self.eigenvalues is not None:
            np.save(output_dir / "eigenvalues.npy", self.eigenvalues)
            
        # Save projections
        if save_projections and self.transformed is not None:
            np.save(output_dir / "projections.npy", self.transformed)
            
            # Save as text for easy plotting
            np.savetxt(
                output_dir / "projections.txt",
                self.transformed[:, :3],
                header="PC1 PC2 PC3"
            )
        
        # Save variance explained
        if self.pca_mda is not None:
            np.savetxt(
                output_dir / "variance_explained.txt",
                np.column_stack([
                    np.arange(len(self.pca_mda.variance)),
                    self.pca_mda.variance,
                    self.pca_mda.cumulated_variance
                ]),
                header="Component Variance Cumulative"
            )
        
        logger.info(f"Saved PCA results to {output_dir}")


def perform_pca(
    universe: mda.Universe,
    selection: str = "protein and name CA",
    n_components: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick PCA analysis.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str, default "protein and name CA"
        Atom selection
    n_components : int, default 10
        Number of components
        
    Returns
    -------
    tuple
        (projections, eigenvalues, eigenvectors)
    """
    pca_analysis = PCAAnalysis(universe, selection)
    results = pca_analysis.run_mda_pca()
    
    return (
        results['transformed'][:, :n_components],
        results['variance'][:n_components],
        results['p_components'][:n_components]
    )


def validate_pca_with_sklearn(
    universe: mda.Universe,
    selection: str = "protein and name CA",
    n_components: int = 10
) -> Dict:
    """
    Validate PCA using sklearn.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str, default "protein and name CA"
        Atom selection
    n_components : int, default 10
        Number of components
        
    Returns
    -------
    dict
        Validation results
    """
    pca_analysis = PCAAnalysis(universe, selection)
    
    # Run both implementations
    pca_analysis.run_mda_pca()
    pca_analysis.run_sklearn_pca(n_components=n_components)
    
    # Validate
    validation = pca_analysis.validate_pca()
    
    return validation