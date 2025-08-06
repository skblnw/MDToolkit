"""Unified trajectory handler for MD simulations."""

import logging
from typing import Optional, Union, List, Tuple
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from pathlib import Path

logger = logging.getLogger(__name__)


class TrajectoryHandler:
    """
    Unified handler for molecular dynamics trajectories.
    
    This class provides a consistent interface for loading and manipulating
    trajectories from various formats using MDAnalysis.
    """
    
    def __init__(
        self,
        topology: Union[str, Path],
        trajectory: Optional[Union[str, Path, List[str]]] = None,
        in_memory: bool = False
    ):
        """
        Initialize trajectory handler.
        
        Parameters
        ----------
        topology : str or Path
            Path to topology file (PDB, GRO, PSF, etc.)
        trajectory : str, Path, or list of str, optional
            Path(s) to trajectory file(s) (XTC, TRR, DCD, etc.)
        in_memory : bool, default False
            Whether to load entire trajectory into memory
        """
        self.topology = Path(topology)
        self.trajectory_files = trajectory
        self.in_memory = in_memory
        
        # Load universe
        if trajectory is None:
            self.universe = mda.Universe(str(self.topology))
            logger.info(f"Loaded topology: {self.topology}")
        else:
            if isinstance(trajectory, (str, Path)):
                trajectory = [str(trajectory)]
            elif isinstance(trajectory, list):
                trajectory = [str(t) for t in trajectory]
                
            self.universe = mda.Universe(str(self.topology), trajectory)
            logger.info(f"Loaded topology: {self.topology} with {len(trajectory)} trajectory file(s)")
            
            if in_memory:
                self.universe.transfer_to_memory()
                logger.info("Trajectory loaded into memory")
    
    def align_trajectory(
        self,
        reference: Optional[mda.Universe] = None,
        selection: str = "protein and name CA",
        ref_frame: int = 0
    ) -> None:
        """
        Align trajectory to reference structure.
        
        Parameters
        ----------
        reference : MDAnalysis.Universe, optional
            Reference universe for alignment. If None, uses first frame
        selection : str, default "protein and name CA"
            Selection string for alignment
        ref_frame : int, default 0
            Reference frame index if reference is None
        """
        if reference is None:
            reference = self.universe
            reference.trajectory[ref_frame]
            
        aligner = align.AlignTraj(
            self.universe,
            reference,
            select=selection,
            in_memory=self.in_memory
        ).run()
        
        logger.info(f"Aligned trajectory using selection: {selection}")
    
    def get_selection(self, selection_string: str) -> mda.AtomGroup:
        """
        Get atom selection from universe.
        
        Parameters
        ----------
        selection_string : str
            MDAnalysis selection string
            
        Returns
        -------
        MDAnalysis.AtomGroup
            Selected atoms
        """
        return self.universe.select_atoms(selection_string)
    
    def iterate_frames(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ):
        """
        Iterate over trajectory frames.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional  
            Ending frame
        step : int, default 1
            Step between frames
            
        Yields
        ------
        MDAnalysis.Timestep
            Trajectory frame
        """
        for ts in self.universe.trajectory[start:stop:step]:
            yield ts
    
    def get_positions(
        self,
        selection: Optional[str] = None,
        frames: Optional[Union[int, List[int], slice]] = None
    ) -> np.ndarray:
        """
        Extract positions for selected atoms and frames.
        
        Parameters
        ----------
        selection : str, optional
            Atom selection string. If None, uses all atoms
        frames : int, list of int, or slice, optional
            Frames to extract. If None, uses all frames
            
        Returns
        -------
        np.ndarray
            Positions array with shape (n_frames, n_atoms, 3)
        """
        atoms = self.universe.atoms if selection is None else self.get_selection(selection)
        
        if frames is None:
            frames = slice(None)
        elif isinstance(frames, int):
            frames = [frames]
            
        positions = []
        for ts in self.universe.trajectory[frames]:
            positions.append(atoms.positions.copy())
            
        return np.array(positions)
    
    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        return len(self.universe.trajectory)
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms in system."""
        return self.universe.atoms.n_atoms
    
    @property
    def time(self) -> np.ndarray:
        """Time values for each frame."""
        return np.array([ts.time for ts in self.universe.trajectory])
    
    def save_selection(
        self,
        selection: str,
        output_file: Union[str, Path],
        frames: Optional[Union[int, List[int], slice]] = None
    ) -> None:
        """
        Save selected atoms to file.
        
        Parameters
        ----------
        selection : str
            Atom selection string
        output_file : str or Path
            Output file path
        frames : int, list of int, or slice, optional
            Frames to save
        """
        atoms = self.get_selection(selection)
        
        with mda.Writer(str(output_file), atoms.n_atoms) as writer:
            if frames is None:
                frames = slice(None)
            elif isinstance(frames, int):
                frames = [frames]
                
            for ts in self.universe.trajectory[frames]:
                writer.write(atoms)
                
        logger.info(f"Saved selection to: {output_file}")