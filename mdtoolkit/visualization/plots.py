"""Comprehensive plotting functions for MD analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path

from .plot_templates import PublicationPlot


def plot_rmsd(
    time: np.ndarray,
    rmsd: Union[np.ndarray, Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    title: str = "RMSD",
    xlabel: str = "Time (ns)",
    ylabel: str = "RMSD (Å)",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot RMSD time series.
    
    Parameters
    ----------
    time : np.ndarray
        Time values
    rmsd : np.ndarray or dict
        RMSD values or dictionary of {label: rmsd}
    labels : list of str, optional
        Labels for multiple RMSD series
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    with PublicationPlot(figsize=figsize) as plot:
        fig, ax = plot.fig, plot.ax
        
        if isinstance(rmsd, dict):
            for label, values in rmsd.items():
                ax.plot(time, values, label=label, linewidth=2)
            ax.legend()
        elif rmsd.ndim == 2:
            if labels is None:
                labels = [f"Selection {i+1}" for i in range(rmsd.shape[1])]
            for i, label in enumerate(labels):
                ax.plot(time, rmsd[:, i], label=label, linewidth=2)
            ax.legend()
        else:
            ax.plot(time, rmsd, linewidth=2, color='#2E86AB')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_rmsf(
    residues: np.ndarray,
    rmsf: np.ndarray,
    title: str = "RMSF by Residue",
    xlabel: str = "Residue",
    ylabel: str = "RMSF (Å)",
    figsize: Tuple[float, float] = (12, 6),
    highlight_regions: Optional[List[Tuple[int, int, str]]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot RMSF by residue.
    
    Parameters
    ----------
    residues : np.ndarray
        Residue numbers
    rmsf : np.ndarray
        RMSF values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    highlight_regions : list of tuples, optional
        Regions to highlight [(start, end, label), ...]
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    with PublicationPlot(figsize=figsize) as plot:
        fig, ax = plot.fig, plot.ax
        
        ax.plot(residues, rmsf, linewidth=2, color='#A23B72')
        ax.fill_between(residues, 0, rmsf, alpha=0.3, color='#A23B72')
        
        # Highlight regions if specified
        if highlight_regions:
            for start, end, label in highlight_regions:
                ax.axvspan(start, end, alpha=0.2, color='orange', label=label)
            ax.legend()
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_correlation_matrix(
    correlation: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot correlation matrix as heatmap.
    
    Parameters
    ----------
    correlation : np.ndarray
        Correlation matrix
    labels : list of str, optional
        Axis labels
    title : str
        Plot title
    cmap : str
        Colormap
    vmin : float
        Minimum value for colormap
    vmax : float
        Maximum value for colormap
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(correlation, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    ax.set_title(title)
    ax.set_xlabel('Residue/Atom Index')
    ax.set_ylabel('Residue/Atom Index')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_pca(
    projections: np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1,
    color_by: Optional[np.ndarray] = None,
    title: str = "PCA Projection",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot PCA projections.
    
    Parameters
    ----------
    projections : np.ndarray
        PCA projections
    pc_x : int, default 0
        PC for x-axis
    pc_y : int, default 1
        PC for y-axis
    color_by : np.ndarray, optional
        Values for coloring points (e.g., time)
    title : str
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    with PublicationPlot(figsize=figsize) as plot:
        fig, ax = plot.fig, plot.ax
        
        if color_by is None:
            color_by = np.arange(len(projections))
        
        scatter = ax.scatter(
            projections[:, pc_x],
            projections[:, pc_y],
            c=color_by,
            cmap='viridis',
            alpha=0.6,
            edgecolors='none'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        if color_by is not None:
            cbar.set_label('Frame/Time', rotation=270, labelpad=20)
        
        ax.set_xlabel(xlabel or f'PC{pc_x + 1}')
        ax.set_ylabel(ylabel or f'PC{pc_y + 1}')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_free_energy_landscape(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 50,
    temperature: float = 300.0,
    title: str = "Free Energy Landscape",
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot free energy landscape from 2D projections.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    bins : int, default 50
        Number of bins for histogram
    temperature : float, default 300.0
        Temperature in Kelvin
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Calculate 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    
    # Convert to probability
    prob = H / H.sum()
    
    # Calculate free energy (in kT units)
    kT = 0.001987 * temperature  # kcal/mol
    with np.errstate(divide='ignore', invalid='ignore'):
        free_energy = -kT * np.log(prob)
        free_energy[np.isinf(free_energy)] = np.nan
    
    # Shift minimum to zero
    min_fe = np.nanmin(free_energy)
    free_energy = free_energy - min_fe
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot as contour
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    contour = ax.contourf(X, Y, free_energy.T, levels=20, cmap='viridis')
    
    # Add contour lines
    ax.contour(X, Y, free_energy.T, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Free Energy (kcal/mol)', rotation=270, labelpad=20)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_contacts(
    time: np.ndarray,
    contacts: np.ndarray,
    title: str = "Contact Analysis",
    xlabel: str = "Time (ns)",
    ylabel: str = "Number of Contacts",
    figsize: Tuple[float, float] = (10, 6),
    running_avg_window: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot contact analysis results.
    
    Parameters
    ----------
    time : np.ndarray
        Time values
    contacts : np.ndarray
        Contact values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    running_avg_window : int, optional
        Window for running average
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    with PublicationPlot(figsize=figsize) as plot:
        fig, ax = plot.fig, plot.ax
        
        # Plot raw data
        ax.plot(time, contacts, alpha=0.5, linewidth=1, label='Raw', color='gray')
        
        # Add running average if requested
        if running_avg_window:
            from ..core.utils import running_average
            avg = running_average(contacts, running_avg_window)
            # Adjust time array for valid convolution
            time_avg = time[:len(avg)]
            ax.plot(time_avg, avg, linewidth=2, label=f'Running avg (w={running_avg_window})',
                   color='#E63946')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if running_avg_window:
            ax.legend()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_timeseries(
    data: Dict[str, np.ndarray],
    time: Optional[np.ndarray] = None,
    title: str = "Time Series Analysis",
    xlabel: str = "Time (ns)",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (12, 8),
    subplots: bool = False,
    save_path: Optional[Union[str, Path]] = None
) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot multiple time series.
    
    Parameters
    ----------
    data : dict
        Dictionary of {label: values}
    time : np.ndarray, optional
        Time values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    subplots : bool, default False
        Whether to create subplots
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    n_series = len(data)
    
    if subplots:
        fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)
        if n_series == 1:
            axes = [axes]
        
        for i, (label, values) in enumerate(data.items()):
            ax = axes[i]
            x = time if time is not None else np.arange(len(values))
            ax.plot(x, values, linewidth=2)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(title)
        
    else:
        with PublicationPlot(figsize=figsize) as plot:
            fig, ax = plot.fig, plot.ax
            axes = ax
            
            for label, values in data.items():
                x = time if time is not None else np.arange(len(values))
                ax.plot(x, values, label=label, linewidth=2)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes