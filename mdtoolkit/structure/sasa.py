"""Solvent accessible surface area (SASA) analysis."""

import numpy as np
import logging
from typing import Optional, Union, Dict
import MDAnalysis as mda

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class SASAAnalysis:
    """
    SASA analysis for MD trajectories.
    
    Uses shrake-rupley algorithm through MDAnalysis.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection: str = "protein",
        probe_radius: float = 1.4
    ):
        """
        Initialize SASA analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection : str
            Selection for SASA calculation
        probe_radius : float
            Probe radius in Angstroms
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection = selection
        self.probe_radius = probe_radius
        self.atoms = self.universe.select_atoms(selection)
        
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> Dict:
        """
        Run SASA analysis.
        
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
        dict
            SASA results
        """
        try:
            from MDAnalysis.analysis.hydrogenbonds.shrake_rupley import shrake_rupley
        except ImportError:
            logger.warning("shrake_rupley not available, using basic calculation")
            return self._run_basic_sasa(start, stop, step)
        
        frames = []
        total_sasa = []
        residue_sasa = []
        
        for ts in self.universe.trajectory[start:stop:step]:
            # Calculate SASA
            sasa = shrake_rupley(
                self.atoms.positions,
                self.atoms.radii,
                probe_radius=self.probe_radius
            )
            
            frames.append(ts.frame)
            total_sasa.append(np.sum(sasa))
            
            # Calculate per-residue SASA
            res_sasa = []
            for residue in self.atoms.residues:
                res_atoms = residue.atoms.intersection(self.atoms)
                res_indices = [self.atoms.indices.tolist().index(a.index) 
                              for a in res_atoms]
                res_sasa.append(np.sum(sasa[res_indices]))
            
            residue_sasa.append(res_sasa)
        
        self.results = {
            'frames': np.array(frames),
            'total_sasa': np.array(total_sasa),
            'residue_sasa': np.array(residue_sasa),
            'mean_sasa': np.mean(total_sasa),
            'std_sasa': np.std(total_sasa)
        }
        
        return self.results
    
    def _run_basic_sasa(
        self,
        start: Optional[int],
        stop: Optional[int],
        step: int
    ) -> Dict:
        """Basic SASA calculation using neighbor counting."""
        frames = []
        total_sasa = []
        
        for ts in self.universe.trajectory[start:stop:step]:
            # Simple approximation based on neighbor counting
            sasa_approx = 0
            
            for atom in self.atoms:
                # Count neighbors within probe distance
                neighbors = self.universe.select_atoms(
                    f"around {2 * self.probe_radius} index {atom.index}"
                )
                
                # Approximate SASA based on neighbor count
                # More neighbors = less exposed surface
                neighbor_factor = max(0, 1 - len(neighbors) / 20)
                atom_sasa = 4 * np.pi * (atom.radius + self.probe_radius)**2
                sasa_approx += atom_sasa * neighbor_factor
            
            frames.append(ts.frame)
            total_sasa.append(sasa_approx)
        
        self.results = {
            'frames': np.array(frames),
            'total_sasa': np.array(total_sasa),
            'mean_sasa': np.mean(total_sasa),
            'std_sasa': np.std(total_sasa)
        }
        
        return self.results
    
    def calculate_buried_surface_area(
        self,
        selection1: str,
        selection2: str
    ) -> np.ndarray:
        """
        Calculate buried surface area between two selections.
        
        Parameters
        ----------
        selection1 : str
            First selection
        selection2 : str
            Second selection
            
        Returns
        -------
        np.ndarray
            Buried surface area over time
        """
        if self.results is None:
            self.run()
        
        group1 = self.universe.select_atoms(selection1)
        group2 = self.universe.select_atoms(selection2)
        complex_sel = f"({selection1}) or ({selection2})"
        
        bsa_values = []
        
        for ts in self.universe.trajectory:
            # Calculate SASA for individual groups
            sasa1 = self._calculate_sasa_for_atoms(group1)
            sasa2 = self._calculate_sasa_for_atoms(group2)
            
            # Calculate SASA for complex
            complex_atoms = self.universe.select_atoms(complex_sel)
            sasa_complex = self._calculate_sasa_for_atoms(complex_atoms)
            
            # BSA = SASA1 + SASA2 - SASA_complex
            bsa = sasa1 + sasa2 - sasa_complex
            bsa_values.append(bsa)
        
        return np.array(bsa_values)
    
    def _calculate_sasa_for_atoms(self, atoms: mda.AtomGroup) -> float:
        """Calculate SASA for specific atoms."""
        # Simple approximation
        total_sasa = 0
        for atom in atoms:
            neighbors = self.universe.select_atoms(
                f"around {2 * self.probe_radius} index {atom.index}"
            )
            neighbor_factor = max(0, 1 - len(neighbors) / 20)
            atom_sasa = 4 * np.pi * (atom.radius + self.probe_radius)**2
            total_sasa += atom_sasa * neighbor_factor
        
        return total_sasa


def calculate_sasa(
    universe: mda.Universe,
    selection: str = "protein",
    probe_radius: float = 1.4
) -> np.ndarray:
    """
    Quick SASA calculation.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str
        Atom selection
    probe_radius : float
        Probe radius
        
    Returns
    -------
    np.ndarray
        SASA values over time
    """
    sasa_analysis = SASAAnalysis(universe, selection, probe_radius)
    results = sasa_analysis.run()
    return results['total_sasa']