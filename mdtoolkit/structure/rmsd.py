"""RMSD and RMSF analysis for molecular dynamics trajectories."""

import numpy as np
import logging
from typing import Optional, Union, List, Dict
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from pathlib import Path

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class RMSDAnalysis:
    """
    Comprehensive RMSD analysis for MD trajectories.
    
    Supports multiple selections, domain-specific alignment,
    and various output formats.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        reference: Optional[Union[TrajectoryHandler, mda.Universe]] = None,
        align_selection: str = "protein and name CA",
        analysis_selections: Optional[Dict[str, str]] = None
    ):
        """
        Initialize RMSD analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        reference : TrajectoryHandler or MDAnalysis.Universe, optional
            Reference structure. If None, uses first frame
        align_selection : str, default "protein and name CA"
            Selection for alignment
        analysis_selections : dict, optional
            Dictionary of {name: selection} for analysis groups
        """
        # Handle trajectory input
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        # Handle reference
        if reference is None:
            self.reference = self.universe
            self.ref_frame = 0
        elif isinstance(reference, TrajectoryHandler):
            self.reference = reference.universe
            self.ref_frame = 0
        else:
            self.reference = reference
            self.ref_frame = 0
            
        self.align_selection = align_selection
        
        # Set up analysis selections
        if analysis_selections is None:
            self.analysis_selections = {"all": align_selection}
        else:
            self.analysis_selections = analysis_selections
            
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        per_residue: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run RMSD analysis.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int, default 1
            Step between frames
        per_residue : bool, default False
            Calculate per-residue RMSD
            
        Returns
        -------
        dict
            Dictionary with RMSD results for each selection
        """
        results = {}
        
        # Set reference frame
        self.reference.trajectory[self.ref_frame]
        
        for name, selection in self.analysis_selections.items():
            logger.info(f"Calculating RMSD for selection '{name}': {selection}")
            
            R = rms.RMSD(
                self.universe,
                self.reference,
                select=self.align_selection,
                groupselections=[selection],
                ref_frame=self.ref_frame
            )
            
            R.run(start=start, stop=stop, step=step)
            
            # Store results
            results[name] = {
                'time': R.rmsd[:, 1],
                'rmsd': R.rmsd[:, 3],  # Group selection RMSD
                'rmsd_align': R.rmsd[:, 2]  # Alignment RMSD
            }
            
            if per_residue:
                results[name]['per_residue'] = self._calculate_per_residue_rmsd(
                    selection, start, stop, step
                )
        
        self.results = results
        return results
    
    def _calculate_per_residue_rmsd(
        self,
        selection: str,
        start: Optional[int],
        stop: Optional[int],
        step: int
    ) -> np.ndarray:
        """Calculate per-residue RMSD."""
        atoms = self.universe.select_atoms(selection)
        ref_atoms = self.reference.select_atoms(selection)
        
        # Get unique residues
        residues = atoms.residues
        n_residues = len(residues)
        
        # Count frames
        frames = list(range(
            start or 0,
            stop or len(self.universe.trajectory),
            step
        ))
        n_frames = len(frames)
        
        # Initialize array
        per_res_rmsd = np.zeros((n_frames, n_residues))
        
        # Calculate per-residue RMSD
        for i, frame_idx in enumerate(frames):
            self.universe.trajectory[frame_idx]
            
            for j, residue in enumerate(residues):
                res_atoms = residue.atoms
                ref_res_atoms = ref_atoms.residues[j].atoms
                
                if len(res_atoms) > 0:
                    per_res_rmsd[i, j] = rms.rmsd(
                        res_atoms.positions,
                        ref_res_atoms.positions
                    )
        
        return per_res_rmsd
    
    def calculate_convergence(
        self,
        selection_name: str = "all",
        window_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Calculate RMSD convergence.
        
        Parameters
        ----------
        selection_name : str, default "all"
            Name of selection to analyze
        window_size : int, default 100
            Window size for running average
            
        Returns
        -------
        dict
            Convergence metrics
        """
        if self.results is None:
            raise RuntimeError("Must run analysis first")
            
        rmsd = self.results[selection_name]['rmsd']
        
        # Running average
        n_windows = len(rmsd) - window_size + 1
        running_avg = np.zeros(n_windows)
        running_std = np.zeros(n_windows)
        
        for i in range(n_windows):
            window_data = rmsd[i:i+window_size]
            running_avg[i] = np.mean(window_data)
            running_std[i] = np.std(window_data)
        
        return {
            'running_average': running_avg,
            'running_std': running_std,
            'convergence_index': running_std / running_avg
        }
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        format: str = "csv"
    ) -> None:
        """
        Save RMSD results to files.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        format : str, default "csv"
            Output format ('csv', 'npy', 'txt')
        """
        if self.results is None:
            raise RuntimeError("Must run analysis first")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, data in self.results.items():
            if format == "csv":
                import pandas as pd
                df = pd.DataFrame({
                    'time': data['time'],
                    'rmsd': data['rmsd'],
                    'rmsd_align': data['rmsd_align']
                })
                df.to_csv(output_dir / f"rmsd_{name}.csv", index=False)
                
            elif format == "npy":
                np.save(
                    output_dir / f"rmsd_{name}.npy",
                    np.column_stack([data['time'], data['rmsd']])
                )
                
            elif format == "txt":
                np.savetxt(
                    output_dir / f"rmsd_{name}.txt",
                    np.column_stack([data['time'], data['rmsd']]),
                    header="time rmsd"
                )
                
        logger.info(f"Saved RMSD results to {output_dir}")


def calculate_rmsd(
    universe: mda.Universe,
    reference: Optional[mda.Universe] = None,
    selection: str = "protein and name CA",
    ref_frame: int = 0
) -> np.ndarray:
    """
    Quick RMSD calculation.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    reference : MDAnalysis.Universe, optional
        Reference universe
    selection : str, default "protein and name CA"
        Selection string
    ref_frame : int, default 0
        Reference frame
        
    Returns
    -------
    np.ndarray
        Array with columns [frame, time, rmsd]
    """
    if reference is None:
        reference = universe
        
    R = rms.RMSD(
        universe,
        reference,
        select=selection,
        ref_frame=ref_frame
    )
    R.run()
    
    return R.rmsd[:, [0, 1, 2]]


def calculate_rmsf(
    universe: mda.Universe,
    selection: str = "protein and name CA",
    average_coordinates: bool = True
) -> np.ndarray:
    """
    Calculate root mean square fluctuation (RMSF).
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str, default "protein and name CA"
        Selection string
    average_coordinates : bool, default True
        Whether to use average coordinates as reference
        
    Returns
    -------
    np.ndarray
        RMSF values for each atom
    """
    atoms = universe.select_atoms(selection)
    
    if average_coordinates:
        # Calculate average positions
        avg_positions = np.zeros_like(atoms.positions)
        for ts in universe.trajectory:
            avg_positions += atoms.positions
        avg_positions /= len(universe.trajectory)
        
        # Calculate RMSF
        rmsf = np.zeros(len(atoms))
        for ts in universe.trajectory:
            diff = atoms.positions - avg_positions
            rmsf += np.sum(diff**2, axis=1)
        
        rmsf = np.sqrt(rmsf / len(universe.trajectory))
        
    else:
        # Use first frame as reference
        universe.trajectory[0]
        ref_positions = atoms.positions.copy()
        
        rmsf = np.zeros(len(atoms))
        for ts in universe.trajectory:
            diff = atoms.positions - ref_positions
            rmsf += np.sum(diff**2, axis=1)
        
        rmsf = np.sqrt(rmsf / len(universe.trajectory))
    
    return rmsf