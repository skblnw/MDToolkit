"""Color palettes and colormaps for MD visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Optional, Union


# Publication-quality color palettes
PALETTES = {
    'default': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C464E'],
    'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F'],
    'science': ['#3B4992', '#EE0000', '#008B45', '#631879', '#008280'],
    'pastel': ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3'],
    'bright': ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00'],
    'dark': ['#1B1B3A', '#2A2D34', '#3C3C3C', '#525252', '#6B6B6B'],
    'diverging': ['#313695', '#4575B4', '#FFFFBF', '#F46D43', '#A50026'],
    'sequential': ['#FFF7EC', '#FEE8C8', '#FDD49E', '#FDBB84', '#FC8D59']
}

# Residue type colors (for structure visualization)
RESIDUE_COLORS = {
    # Hydrophobic
    'ALA': '#C8C8C8',
    'VAL': '#C8C8C8',
    'ILE': '#C8C8C8',
    'LEU': '#C8C8C8',
    'MET': '#C8C8C8',
    'PHE': '#C8C8C8',
    'TRP': '#C8C8C8',
    'PRO': '#C8C8C8',
    # Polar
    'SER': '#FA9600',
    'THR': '#FA9600',
    'CYS': '#FA9600',
    'TYR': '#FA9600',
    'ASN': '#FA9600',
    'GLN': '#FA9600',
    # Basic
    'LYS': '#145AFF',
    'ARG': '#145AFF',
    'HIS': '#145AFF',
    # Acidic
    'ASP': '#E60A0A',
    'GLU': '#E60A0A',
    # Special
    'GLY': '#EBEBEB'
}

# Secondary structure colors
SECONDARY_STRUCTURE_COLORS = {
    'helix': '#E41A1C',
    'sheet': '#377EB8',
    'turn': '#4DAF4A',
    'coil': '#984EA3',
    'bridge': '#FF7F00',
    '3-10helix': '#FFFF33',
    'pi-helix': '#A65628'
}


def get_color_palette(
    name: str = 'default',
    n_colors: Optional[int] = None
) -> List[str]:
    """
    Get color palette by name.
    
    Parameters
    ----------
    name : str
        Palette name
    n_colors : int, optional
        Number of colors to return
        
    Returns
    -------
    list
        List of color hex codes
    """
    if name not in PALETTES:
        raise ValueError(f"Unknown palette: {name}. Available: {list(PALETTES.keys())}")
    
    palette = PALETTES[name].copy()
    
    if n_colors:
        if n_colors <= len(palette):
            return palette[:n_colors]
        else:
            # Cycle through colors if more needed
            extended = []
            for i in range(n_colors):
                extended.append(palette[i % len(palette)])
            return extended
    
    return palette


def get_colormap(
    name: str = 'viridis',
    n_colors: int = 256,
    vmin: float = 0,
    vmax: float = 1,
    center: Optional[float] = None
) -> mcolors.LinearSegmentedColormap:
    """
    Get or create a colormap.
    
    Parameters
    ----------
    name : str
        Colormap name or 'custom'
    n_colors : int
        Number of colors
    vmin : float
        Minimum value
    vmax : float
        Maximum value
    center : float, optional
        Center value for diverging colormaps
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Colormap object
    """
    if name == 'correlation':
        # Blue-white-red for correlation matrices
        colors = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8',
                 '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        n_bins = len(colors)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'correlation', colors, N=n_bins
        )
        
    elif name == 'rmsd':
        # Green to red for RMSD
        colors = ['#00FF00', '#FFFF00', '#FF0000']
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'rmsd', colors, N=n_colors
        )
        
    elif name == 'contact':
        # White to dark blue for contacts
        colors = ['#FFFFFF', '#E0F3DB', '#CCEBC5', '#A8DDB5', '#7BCCC4',
                 '#4EB3D3', '#2B8CBE', '#0868AC', '#084081']
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'contact', colors, N=n_colors
        )
        
    elif name == 'energy':
        # Custom free energy landscape
        colors = ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E',
                 '#1F9E89', '#35B779', '#6DCD59', '#B4DE2C', '#FDE725']
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'energy', colors, N=n_colors
        )
        
    else:
        # Use matplotlib colormap
        cmap = plt.get_cmap(name)
    
    # Apply normalization if center is specified
    if center is not None:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        return cmap, norm
    
    return cmap


def color_by_property(
    values: np.ndarray,
    cmap_name: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> np.ndarray:
    """
    Color values by property.
    
    Parameters
    ----------
    values : np.ndarray
        Property values
    cmap_name : str
        Colormap name
    vmin : float, optional
        Minimum value for scaling
    vmax : float, optional
        Maximum value for scaling
        
    Returns
    -------
    np.ndarray
        RGB colors array
    """
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    # Normalize values
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = get_colormap(cmap_name)
    if isinstance(cmap, tuple):
        cmap = cmap[0]
    
    # Map values to colors
    colors = cmap(norm(values))
    
    return colors


def create_custom_colormap(
    colors: List[str],
    name: str = 'custom',
    n_colors: int = 256
) -> mcolors.LinearSegmentedColormap:
    """
    Create custom colormap from color list.
    
    Parameters
    ----------
    colors : list
        List of colors (hex, names, or RGB)
    name : str
        Colormap name
    n_colors : int
        Number of colors in colormap
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap
    """
    return mcolors.LinearSegmentedColormap.from_list(
        name, colors, N=n_colors
    )


def get_residue_colors(
    residues: List[str],
    color_scheme: str = 'type'
) -> List[str]:
    """
    Get colors for residues.
    
    Parameters
    ----------
    residues : list
        List of residue names
    color_scheme : str
        'type' for residue type, 'hydrophobicity', etc.
        
    Returns
    -------
    list
        List of colors
    """
    colors = []
    
    for res in residues:
        if color_scheme == 'type':
            color = RESIDUE_COLORS.get(res.upper(), '#808080')
        elif color_scheme == 'hydrophobicity':
            # Hydrophobicity scale
            hydrophobic = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']
            if res.upper() in hydrophobic:
                color = '#0000FF'  # Blue for hydrophobic
            else:
                color = '#FF0000'  # Red for hydrophilic
        else:
            color = '#808080'  # Default gray
            
        colors.append(color)
    
    return colors