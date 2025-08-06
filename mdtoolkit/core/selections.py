"""Atom selection utilities and common selection definitions."""

from typing import Dict, Optional, Union
import MDAnalysis as mda
import logging

logger = logging.getLogger(__name__)


class SelectionParser:
    """
    Parser for common atom selections in biomolecular systems.
    
    Provides predefined selections and utilities for building
    complex selection strings.
    """
    
    # Common predefined selections
    SELECTIONS = {
        "backbone": "backbone",
        "ca": "protein and name CA",
        "heavy": "not name H*",
        "protein": "protein",
        "nucleic": "nucleic",
        "lipid": "resname POPC POPE POPS DOPC DOPE DOPS",
        "water": "resname WAT HOH TIP3 SPC",
        "ions": "resname NA CL K MG CA ZN",
        "solvent": "resname WAT HOH TIP3 SPC or resname NA CL K MG CA ZN",
    }
    
    def __init__(self):
        """Initialize selection parser."""
        self.custom_selections = {}
        
    def add_custom_selection(self, name: str, selection: str) -> None:
        """
        Add a custom selection definition.
        
        Parameters
        ----------
        name : str
            Name for the selection
        selection : str
            MDAnalysis selection string
        """
        self.custom_selections[name] = selection
        logger.debug(f"Added custom selection '{name}': {selection}")
    
    def get_selection(self, name: str) -> str:
        """
        Get selection string by name.
        
        Parameters
        ----------
        name : str
            Selection name
            
        Returns
        -------
        str
            Selection string
            
        Raises
        ------
        KeyError
            If selection name not found
        """
        if name in self.custom_selections:
            return self.custom_selections[name]
        elif name in self.SELECTIONS:
            return self.SELECTIONS[name]
        else:
            raise KeyError(f"Selection '{name}' not found")
    
    @staticmethod
    def combine_selections(
        selections: list,
        operator: str = "or"
    ) -> str:
        """
        Combine multiple selection strings.
        
        Parameters
        ----------
        selections : list of str
            Selection strings to combine
        operator : str, default "or"
            Logical operator ("or", "and", "not")
            
        Returns
        -------
        str
            Combined selection string
        """
        if operator not in ["or", "and"]:
            raise ValueError(f"Invalid operator: {operator}")
            
        return f" {operator} ".join(f"({sel})" for sel in selections)
    
    @staticmethod
    def residue_range(
        start: int,
        end: int,
        chain: Optional[str] = None
    ) -> str:
        """
        Create selection for residue range.
        
        Parameters
        ----------
        start : int
            Starting residue number
        end : int
            Ending residue number
        chain : str, optional
            Chain/segment identifier
            
        Returns
        -------
        str
            Selection string
        """
        selection = f"resid {start}:{end}"
        if chain:
            selection = f"({selection}) and segid {chain}"
        return selection
    
    @staticmethod
    def within_distance(
        selection: str,
        distance: float,
        target: str = "protein"
    ) -> str:
        """
        Select atoms within distance of target.
        
        Parameters
        ----------
        selection : str
            Base selection
        distance : float
            Distance cutoff in Angstroms
        target : str, default "protein"
            Target selection
            
        Returns
        -------
        str
            Selection string
        """
        return f"({selection}) and around {distance} ({target})"


def get_selection(
    universe: mda.Universe,
    selection: Union[str, Dict[str, Union[str, int, float]]]
) -> mda.AtomGroup:
    """
    Get atom selection from universe with enhanced syntax.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe to select from
    selection : str or dict
        Selection specification
        
    Returns
    -------
    MDAnalysis.AtomGroup
        Selected atoms
        
    Examples
    --------
    >>> # Simple string selection
    >>> atoms = get_selection(u, "protein and name CA")
    
    >>> # Dictionary-based selection
    >>> atoms = get_selection(u, {
    ...     "base": "protein",
    ...     "resid_range": [10, 50],
    ...     "chain": "A"
    ... })
    """
    if isinstance(selection, str):
        return universe.select_atoms(selection)
    
    elif isinstance(selection, dict):
        sel_parts = []
        
        if "base" in selection:
            sel_parts.append(selection["base"])
            
        if "resid_range" in selection:
            start, end = selection["resid_range"]
            sel_parts.append(f"resid {start}:{end}")
            
        if "chain" in selection:
            sel_parts.append(f"segid {selection['chain']}")
            
        if "around" in selection:
            distance = selection.get("distance", 5.0)
            target = selection.get("target", "protein")
            base = " and ".join(sel_parts) if sel_parts else "all"
            return universe.select_atoms(
                f"({base}) and around {distance} ({target})"
            )
        
        if not sel_parts:
            raise ValueError("No valid selection criteria provided")
            
        selection_string = " and ".join(sel_parts)
        return universe.select_atoms(selection_string)
    
    else:
        raise TypeError(f"Invalid selection type: {type(selection)}")