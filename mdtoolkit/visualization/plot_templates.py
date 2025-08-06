"""Publication-ready plot templates for MD analysis."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Optional, Tuple


class PlotTemplate:
    """Base class for plot templates."""
    
    DEFAULT_PARAMS = {
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.linewidth': 1.5,
        'axes.xmargin': 0.02,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'xtick.major.size': 8,
        'xtick.major.width': 1.5,
        'ytick.labelsize': 12,
        'ytick.major.size': 8,
        'ytick.major.width': 1.5,
        'text.usetex': False,
        'figure.figsize': [8, 6],
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    }
    
    def __init__(self, style: Optional[str] = None, **kwargs):
        """
        Initialize plot template.
        
        Parameters
        ----------
        style : str, optional
            Matplotlib style
        **kwargs
            Additional rcParams
        """
        self.original_params = mpl.rcParams.copy()
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(kwargs)
        
        if style:
            plt.style.use(style)
        
        mpl.rcParams.update(self.params)
    
    def reset(self):
        """Reset to original parameters."""
        mpl.rcParams.update(self.original_params)


class PublicationPlot:
    """
    Context manager for publication-quality plots.
    
    Examples
    --------
    >>> with PublicationPlot(figsize=(10, 6)) as plot:
    ...     plot.ax.plot(x, y)
    ...     plot.ax.set_xlabel('Time (ns)')
    ...     plot.save('figure.png')
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (8, 6),
        style: str = 'seaborn-v0_8-whitegrid',
        dpi: int = 100,
        **kwargs
    ):
        """
        Initialize publication plot.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        style : str
            Plot style
        dpi : int
            Figure DPI
        **kwargs
            Additional parameters
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.kwargs = kwargs
        self.fig = None
        self.ax = None
        
    def __enter__(self):
        """Enter context."""
        # Set style
        if self.style:
            plt.style.use(self.style)
        
        # Update parameters
        params = {
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'lines.linewidth': 2.5,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 10,
            'ytick.major.size': 10,
        }
        params.update(self.kwargs)
        
        for key, value in params.items():
            mpl.rcParams[key] = value
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        plt.tight_layout()
        
    def save(self, filename: str, dpi: int = 300, **kwargs):
        """
        Save figure.
        
        Parameters
        ----------
        filename : str
            Output filename
        dpi : int
            Output DPI
        **kwargs
            Additional savefig parameters
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
    
    def set_spine_visibility(self, top=False, right=False, left=True, bottom=True):
        """Set spine visibility."""
        self.ax.spines['top'].set_visible(top)
        self.ax.spines['right'].set_visible(right)
        self.ax.spines['left'].set_visible(left)
        self.ax.spines['bottom'].set_visible(bottom)


def set_publication_style():
    """Set global publication style."""
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.linewidth': 2,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'legend.fontsize': 14,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'xtick.labelsize': 14,
        'xtick.major.size': 10,
        'xtick.major.width': 2,
        'xtick.minor.size': 5,
        'xtick.minor.width': 1,
        'ytick.labelsize': 14,
        'ytick.major.size': 10,
        'ytick.major.width': 2,
        'ytick.minor.size': 5,
        'ytick.minor.width': 1,
        'figure.figsize': [10, 8],
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    }
    
    mpl.rcParams.update(params)