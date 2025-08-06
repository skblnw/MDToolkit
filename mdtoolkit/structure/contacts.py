"""Contact analysis for molecular dynamics trajectories."""

import numpy as np
import logging
from typing import Optional, Union, List, Dict, Tuple
import MDAnalysis as mda
from MDAnalysis.analysis import contacts, distances, hydrogenbonds
from pathlib import Path

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class ContactAnalysis:
    """
    General contact analysis between atom groups.
    
    Calculates contacts based on distance cutoffs and
    provides various analysis methods.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection1: str,
        selection2: str,
        cutoff: float = 5.0
    ):
        """
        Initialize contact analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection1 : str
            First selection
        selection2 : str
            Second selection
        cutoff : float, default 5.0
            Distance cutoff in Angstroms
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection1 = selection1
        self.selection2 = selection2
        self.cutoff = cutoff
        
        self.group1 = self.universe.select_atoms(selection1)
        self.group2 = self.universe.select_atoms(selection2)
        
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        detailed: bool = False
    ) -> Dict:
        """
        Run contact analysis.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int, default 1
            Step between frames
        detailed : bool, default False
            Store detailed contact information
            
        Returns
        -------
        dict
            Contact analysis results
        """
        frames = []
        n_contacts = []
        contact_pairs = [] if detailed else None
        
        for ts in self.universe.trajectory[start:stop:step]:
            # Calculate distances
            dist_array = distances.distance_array(
                self.group1.positions,
                self.group2.positions
            )
            
            # Find contacts
            contact_mask = dist_array < self.cutoff
            n_cont = np.sum(contact_mask)
            
            frames.append(ts.frame)
            n_contacts.append(n_cont)
            
            if detailed:
                # Get indices of contacting atoms
                idx1, idx2 = np.where(contact_mask)
                pairs = [(self.group1[i].index, self.group2[j].index)
                        for i, j in zip(idx1, idx2)]
                contact_pairs.append(pairs)
        
        self.results = {
            'frames': np.array(frames),
            'n_contacts': np.array(n_contacts),
            'contact_pairs': contact_pairs,
            'cutoff': self.cutoff
        }
        
        return self.results
    
    def calculate_contact_map(self) -> np.ndarray:
        """
        Calculate average contact map.
        
        Returns
        -------
        np.ndarray
            Contact probability matrix
        """
        if self.results is None:
            raise RuntimeError("Must run analysis first")
            
        # Initialize contact map
        n1 = len(self.group1)
        n2 = len(self.group2)
        contact_map = np.zeros((n1, n2))
        
        # Calculate contact frequency
        for ts in self.universe.trajectory:
            dist_array = distances.distance_array(
                self.group1.positions,
                self.group2.positions
            )
            contact_map += (dist_array < self.cutoff).astype(float)
        
        # Normalize by number of frames
        contact_map /= len(self.universe.trajectory)
        
        return contact_map
    
    def get_persistent_contacts(
        self,
        persistence_cutoff: float = 0.5
    ) -> List[Tuple[int, int]]:
        """
        Identify persistent contacts.
        
        Parameters
        ----------
        persistence_cutoff : float, default 0.5
            Minimum fraction of frames for persistent contact
            
        Returns
        -------
        list
            List of persistent contact pairs (indices)
        """
        contact_map = self.calculate_contact_map()
        
        # Find persistent contacts
        persistent = np.where(contact_map >= persistence_cutoff)
        
        pairs = [(self.group1[i].index, self.group2[j].index)
                for i, j in zip(persistent[0], persistent[1])]
        
        return pairs


class NativeContacts:
    """
    Native contact analysis for protein folding studies.
    
    Uses soft-switching function for smooth contact definition.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        reference: Optional[Union[TrajectoryHandler, mda.Universe]] = None,
        selection: str = "protein and name CA",
        radius: float = 4.5,
        beta: float = 5.0,
        lambda_constant: float = 1.5
    ):
        """
        Initialize native contact analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        reference : TrajectoryHandler or MDAnalysis.Universe, optional
            Reference structure for native contacts
        selection : str, default "protein and name CA"
            Selection for contact calculation
        radius : float, default 4.5
            Contact radius in Angstroms
        beta : float, default 5.0
            Softness parameter
        lambda_constant : float, default 1.5
            Lambda parameter for soft cutoff
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        if reference is None:
            self.reference = self.universe
        elif isinstance(reference, TrajectoryHandler):
            self.reference = reference.universe
        else:
            self.reference = reference
            
        self.selection = selection
        self.radius = radius
        self.beta = beta
        self.lambda_constant = lambda_constant
        
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> Dict:
        """
        Run native contact analysis.
        
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
        dict
            Native contact results
        """
        # Set up reference groups
        refgroup = self.reference.select_atoms(self.selection)
        selgroup = self.universe.select_atoms(self.selection)
        
        # Run analysis
        nc = contacts.Contacts(
            self.universe,
            selection=(self.selection, self.selection),
            refgroup=(refgroup, refgroup),
            radius=self.radius,
            method='soft_cut',
            kwargs={
                'beta': self.beta,
                'lambda_constant': self.lambda_constant
            }
        )
        
        nc.run(start=start, stop=stop, step=step)
        
        self.results = {
            'time': nc.times,
            'q': nc.timeseries[:, 1],  # Fraction of native contacts
            'n_contacts': nc.timeseries[:, 0],  # Number of contacts
        }
        
        return self.results


class HydrogenBonds:
    """
    Hydrogen bond analysis for MD trajectories.
    
    Identifies and tracks hydrogen bonds based on geometric criteria.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        donors: Optional[str] = None,
        acceptors: Optional[str] = None,
        distance_cutoff: float = 3.5,
        angle_cutoff: float = 150.0
    ):
        """
        Initialize hydrogen bond analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        donors : str, optional
            Donor selection. If None, uses protein donors
        acceptors : str, optional
            Acceptor selection. If None, uses protein acceptors
        distance_cutoff : float, default 3.5
            Maximum D-A distance in Angstroms
        angle_cutoff : float, default 150.0
            Minimum D-H-A angle in degrees
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        # Default selections for protein
        if donors is None:
            donors = "protein and (name N or name N*)"
        if acceptors is None:
            acceptors = "protein and (name O or name O*)"
            
        self.donors = donors
        self.acceptors = acceptors
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
        
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        update_selections: bool = True
    ) -> Dict:
        """
        Run hydrogen bond analysis.
        
        Parameters
        ----------
        start : int, optional
            Starting frame
        stop : int, optional
            Ending frame
        step : int, default 1
            Step between frames
        update_selections : bool, default True
            Update selections each frame
            
        Returns
        -------
        dict
            Hydrogen bond results
        """
        # Set up hydrogen bond analysis
        hbonds = hydrogenbonds.HydrogenBondAnalysis(
            self.universe,
            donors_sel=self.donors,
            acceptors_sel=self.acceptors,
            d_a_cutoff=self.distance_cutoff,
            d_h_a_angle_cutoff=self.angle_cutoff,
            update_selections=update_selections
        )
        
        hbonds.run(start=start, stop=stop, step=step)
        
        # Process results
        counts = hbonds.count_by_time()
        
        self.results = {
            'time': counts[:, 0],
            'n_hbonds': counts[:, 1],
            'hbonds': hbonds.results.hbonds,  # Detailed H-bond information
        }
        
        return self.results
    
    def get_persistent_hbonds(
        self,
        persistence_cutoff: float = 0.5
    ) -> List:
        """
        Identify persistent hydrogen bonds.
        
        Parameters
        ----------
        persistence_cutoff : float, default 0.5
            Minimum fraction of frames for persistence
            
        Returns
        -------
        list
            Persistent hydrogen bonds
        """
        if self.results is None:
            raise RuntimeError("Must run analysis first")
            
        # Count occurrences of each H-bond
        hbond_counts = {}
        total_frames = len(self.results['time'])
        
        for hbond in self.results['hbonds']:
            # Create unique identifier for H-bond
            key = (hbond[1], hbond[3])  # Donor and acceptor indices
            
            if key not in hbond_counts:
                hbond_counts[key] = 0
            hbond_counts[key] += 1
        
        # Find persistent H-bonds
        persistent = []
        for key, count in hbond_counts.items():
            if count / total_frames >= persistence_cutoff:
                persistent.append({
                    'donor': key[0],
                    'acceptor': key[1],
                    'persistence': count / total_frames
                })
        
        return persistent


class SaltBridges:
    """
    Salt bridge analysis for charged residue interactions.
    """
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        cations: Optional[str] = None,
        anions: Optional[str] = None,
        distance_cutoff: float = 4.0
    ):
        """
        Initialize salt bridge analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        cations : str, optional
            Cation selection. If None, uses basic residues
        anions : str, optional
            Anion selection. If None, uses acidic residues
        distance_cutoff : float, default 4.0
            Maximum distance in Angstroms
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        # Default selections for charged residues
        if cations is None:
            cations = "protein and (resname ARG LYS) and name NH* NZ"
        if anions is None:
            anions = "protein and (resname ASP GLU) and name OE* OD*"
            
        self.cations = self.universe.select_atoms(cations)
        self.anions = self.universe.select_atoms(anions)
        self.distance_cutoff = distance_cutoff
        
        self.results = None
        
    @timing_decorator
    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1
    ) -> Dict:
        """
        Run salt bridge analysis.
        
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
        dict
            Salt bridge results
        """
        frames = []
        n_bridges = []
        bridge_pairs = []
        
        for ts in self.universe.trajectory[start:stop:step]:
            # Calculate distances between charged groups
            dist_array = distances.distance_array(
                self.cations.positions,
                self.anions.positions
            )
            
            # Find salt bridges
            bridge_mask = dist_array < self.distance_cutoff
            n_sb = np.sum(bridge_mask)
            
            frames.append(ts.frame)
            n_bridges.append(n_sb)
            
            # Get salt bridge pairs
            idx1, idx2 = np.where(bridge_mask)
            pairs = [(self.cations[i].resid, self.anions[j].resid)
                    for i, j in zip(idx1, idx2)]
            bridge_pairs.append(pairs)
        
        self.results = {
            'frames': np.array(frames),
            'n_bridges': np.array(n_bridges),
            'bridge_pairs': bridge_pairs,
            'cutoff': self.distance_cutoff
        }
        
        return self.results


def calculate_contact_map(
    universe: mda.Universe,
    selection1: str = "protein and name CA",
    selection2: Optional[str] = None,
    cutoff: float = 8.0
) -> np.ndarray:
    """
    Quick contact map calculation.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection1 : str
        First selection
    selection2 : str, optional
        Second selection. If None, uses selection1
    cutoff : float, default 8.0
        Distance cutoff
        
    Returns
    -------
    np.ndarray
        Average contact map
    """
    if selection2 is None:
        selection2 = selection1
        
    group1 = universe.select_atoms(selection1)
    group2 = universe.select_atoms(selection2)
    
    contact_map = np.zeros((len(group1), len(group2)))
    
    for ts in universe.trajectory:
        dist = distances.distance_array(
            group1.positions,
            group2.positions
        )
        contact_map += (dist < cutoff).astype(float)
    
    contact_map /= len(universe.trajectory)
    
    return contact_map