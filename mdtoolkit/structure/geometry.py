"""Geometric analysis for MD trajectories."""

import numpy as np
import logging
from typing import Optional, Union, Tuple, List
import MDAnalysis as mda
from MDAnalysis.analysis import distances

from ..core.trajectory import TrajectoryHandler
from ..core.utils import timing_decorator

logger = logging.getLogger(__name__)


class RadiusOfGyration:
    """Calculate radius of gyration over trajectory."""
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection: str = "protein"
    ):
        """
        Initialize radius of gyration analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection : str
            Atom selection
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection = selection
        self.atoms = self.universe.select_atoms(selection)
        self.results = None
        
    @timing_decorator
    def run(self) -> np.ndarray:
        """
        Calculate radius of gyration.
        
        Returns
        -------
        np.ndarray
            Radius of gyration over time
        """
        rgyr = []
        
        for ts in self.universe.trajectory:
            rgyr.append(self.atoms.radius_of_gyration())
        
        self.results = np.array(rgyr)
        return self.results


class EndToEndDistance:
    """Calculate end-to-end distance for polymers."""
    
    def __init__(
        self,
        trajectory: Union[TrajectoryHandler, mda.Universe],
        selection: str = "protein and name CA"
    ):
        """
        Initialize end-to-end distance analysis.
        
        Parameters
        ----------
        trajectory : TrajectoryHandler or MDAnalysis.Universe
            Trajectory to analyze
        selection : str
            Atom selection
        """
        if isinstance(trajectory, TrajectoryHandler):
            self.universe = trajectory.universe
        else:
            self.universe = trajectory
            
        self.selection = selection
        self.atoms = self.universe.select_atoms(selection)
        self.results = None
        
    @timing_decorator
    def run(self) -> np.ndarray:
        """
        Calculate end-to-end distance.
        
        Returns
        -------
        np.ndarray
            End-to-end distances over time
        """
        distances = []
        
        for ts in self.universe.trajectory:
            # Get first and last atoms
            first = self.atoms[0].position
            last = self.atoms[-1].position
            
            # Calculate distance
            dist = np.linalg.norm(last - first)
            distances.append(dist)
        
        self.results = np.array(distances)
        return self.results


def calculate_angle(
    universe: mda.Universe,
    atom1: Union[int, mda.Atom],
    atom2: Union[int, mda.Atom],
    atom3: Union[int, mda.Atom]
) -> np.ndarray:
    """
    Calculate angle between three atoms over trajectory.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    atom1, atom2, atom3 : int or Atom
        Atoms defining the angle (atom2 is vertex)
        
    Returns
    -------
    np.ndarray
        Angles in degrees over time
    """
    # Convert indices to atoms if needed
    if isinstance(atom1, int):
        atom1 = universe.atoms[atom1]
    if isinstance(atom2, int):
        atom2 = universe.atoms[atom2]
    if isinstance(atom3, int):
        atom3 = universe.atoms[atom3]
    
    angles = []
    
    for ts in universe.trajectory:
        # Get positions
        pos1 = atom1.position
        pos2 = atom2.position
        pos3 = atom3.position
        
        # Calculate vectors
        vec1 = pos1 - pos2
        vec2 = pos3 - pos2
        
        # Calculate angle
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        angles.append(angle)
    
    return np.array(angles)


def calculate_dihedral(
    universe: mda.Universe,
    atom1: Union[int, mda.Atom],
    atom2: Union[int, mda.Atom],
    atom3: Union[int, mda.Atom],
    atom4: Union[int, mda.Atom]
) -> np.ndarray:
    """
    Calculate dihedral angle over trajectory.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    atom1, atom2, atom3, atom4 : int or Atom
        Atoms defining the dihedral
        
    Returns
    -------
    np.ndarray
        Dihedral angles in degrees over time
    """
    # Convert indices to atoms if needed
    if isinstance(atom1, int):
        atom1 = universe.atoms[atom1]
    if isinstance(atom2, int):
        atom2 = universe.atoms[atom2]
    if isinstance(atom3, int):
        atom3 = universe.atoms[atom3]
    if isinstance(atom4, int):
        atom4 = universe.atoms[atom4]
    
    dihedrals = []
    
    for ts in universe.trajectory:
        # Get positions
        pos1 = atom1.position
        pos2 = atom2.position
        pos3 = atom3.position
        pos4 = atom4.position
        
        # Calculate vectors
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3
        
        # Calculate dihedral using cross products
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Normalize
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # Calculate angle
        m1 = np.cross(n1, b2 / np.linalg.norm(b2))
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        dihedral = np.arctan2(y, x) * 180 / np.pi
        dihedrals.append(dihedral)
    
    return np.array(dihedrals)


def calculate_ramachandran(
    universe: mda.Universe,
    selection: str = "protein"
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calculate Ramachandran angles (phi, psi) for protein.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory universe
    selection : str
        Protein selection
        
    Returns
    -------
    tuple
        (phi_angles, psi_angles) for each residue over time
    """
    protein = universe.select_atoms(selection)
    
    # Get residues (excluding terminals)
    residues = protein.residues[1:-1]
    
    phi_angles = []
    psi_angles = []
    
    for res in residues:
        phi_list = []
        psi_list = []
        
        # Get atoms for phi and psi
        try:
            # Phi: C(-1) - N - CA - C
            c_prev = universe.select_atoms(f"resid {res.resid-1} and name C")[0]
            n = res.atoms.select_atoms("name N")[0]
            ca = res.atoms.select_atoms("name CA")[0]
            c = res.atoms.select_atoms("name C")[0]
            
            # Psi: N - CA - C - N(+1)
            n_next = universe.select_atoms(f"resid {res.resid+1} and name N")[0]
            
            # Calculate over trajectory
            for ts in universe.trajectory:
                phi = calculate_dihedral_single(
                    c_prev.position, n.position, ca.position, c.position
                )
                psi = calculate_dihedral_single(
                    n.position, ca.position, c.position, n_next.position
                )
                
                phi_list.append(phi)
                psi_list.append(psi)
            
            phi_angles.append(np.array(phi_list))
            psi_angles.append(np.array(psi_list))
            
        except (IndexError, AttributeError):
            # Skip if atoms not found
            continue
    
    return phi_angles, psi_angles


def calculate_dihedral_single(
    pos1: np.ndarray,
    pos2: np.ndarray,
    pos3: np.ndarray,
    pos4: np.ndarray
) -> float:
    """Calculate dihedral angle from four positions."""
    b1 = pos2 - pos1
    b2 = pos3 - pos2
    b3 = pos4 - pos3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.arctan2(y, x) * 180 / np.pi