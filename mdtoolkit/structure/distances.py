"""Distance calculations for MD trajectories."""

import numpy as np
import logging
from typing import Optional, Union, List, Tuple
import MDAnalysis as mda
from MDAnalysis.analysis import distances

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class DistanceAnalysis:
    """
    Distance analysis between atom groups.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection1: str,
        selection2: Optional[str] = None
    ):
        """
        Initialize distance analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection1 : str
            First selection
        selection2 : str, optional
            Second selection. If None, calculates within selection1
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection1 = selection1
        self.selection2 = selection2 or selection1
        
        self.group1 = self.universe.select_atoms(selection1)
        self.group2 = self.universe.select_atoms(selection2)
        
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> dict:
        """
        Run distance analysis.
        
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
            Distance analysis results
        """
        frames = []
        mean_distances = []
        min_distances = []
        max_distances = []
        all_distances = []
        
        for ts in self.universe.trajectory[start:stop:step]:
            # Calculate distance array
            dist_array = distances.distance_array(
                self.group1.positions,
                self.group2.positions
            )
            
            frames.append(ts.frame)
            mean_distances.append(np.mean(dist_array))
            min_distances.append(np.min(dist_array))
            max_distances.append(np.max(dist_array))
            all_distances.append(dist_array)
        
        self.results = {
            'frames': np.array(frames),
            'mean': np.array(mean_distances),
            'min': np.array(min_distances),
            'max': np.array(max_distances),
            'all_distances': all_distances
        }
        
        return self.results
    
    def calculate_com_distance(self) -> np.ndarray:
        """
        Calculate center of mass distance.
        
        Returns
        -------
        np.ndarray
            COM distances over time
        """
        com_distances = []
        
        for ts in self.universe.trajectory:
            com1 = self.group1.center_of_mass()
            com2 = self.group2.center_of_mass()
            dist = np.linalg.norm(com1 - com2)
            com_distances.append(dist)
        
        return np.array(com_distances)
    
    def calculate_mindist(self) -> Tuple[np.ndarray, List]:
        """
        Calculate minimum distance and contacting atoms.
        
        Returns
        -------
        tuple
            (min_distances, contact_pairs)
        """
        min_distances = []
        contact_pairs = []
        
        for ts in self.universe.trajectory:
            dist_array = distances.distance_array(
                self.group1.positions,
                self.group2.positions
            )
            
            min_idx = np.unravel_index(np.argmin(dist_array), dist_array.shape)
            min_dist = dist_array[min_idx]
            
            min_distances.append(min_dist)
            contact_pairs.append((
                self.group1[min_idx[0]].index,
                self.group2[min_idx[1]].index
            ))
        
        return np.array(min_distances), contact_pairs


def calculate_distances(
    universe: mda.Universe,
    pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Calculate distances between specific atom pairs.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    pairs : list of tuples
        List of (atom1_index, atom2_index) pairs
        
    Returns
    -------
    np.ndarray
        Distance array (n_frames, n_pairs)
    """
    n_frames = len(universe.trajectory)
    n_pairs = len(pairs)
    distances_array = np.zeros((n_frames, n_pairs))
    
    for i, ts in enumerate(universe.trajectory):
        for j, (idx1, idx2) in enumerate(pairs):
            atom1 = universe.atoms[idx1]
            atom2 = universe.atoms[idx2]
            dist = np.linalg.norm(atom1.position - atom2.position)
            distances_array[i, j] = dist
    
    return distances_array