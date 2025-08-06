"""Potential of Mean Force (PMF) analysis module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from scipy import stats, optimize

logger = logging.getLogger(__name__)


class PMFAnalysis:
    """PMF analysis from umbrella sampling or metadynamics simulations.
    
    Implements WHAM (Weighted Histogram Analysis Method) and other
    methods for calculating PMF profiles from biased simulations.
    """
    
    def __init__(self, temperature: float = 300.0):
        """Initialize PMF analysis.
        
        Args:
            temperature: Temperature in Kelvin (default: 300K)
        """
        self.temperature = temperature
        self.kT = 0.0019872041 * temperature  # kT in kcal/mol
        self.beta = 1.0 / self.kT
        self.windows = []
        self.pmf_profile = None
        self.convergence_data = {}
        
    def add_window(
        self,
        reaction_coord: np.ndarray,
        center: float,
        force_constant: float,
        bias_potential: Optional[np.ndarray] = None
    ):
        """Add umbrella sampling window data.
        
        Args:
            reaction_coord: Reaction coordinate values
            center: Umbrella center position
            force_constant: Force constant (kcal/mol/Å²)
            bias_potential: Optional pre-computed bias potential
        """
        if bias_potential is None:
            # Harmonic bias: V = 0.5 * k * (x - x0)²
            bias_potential = 0.5 * force_constant * (reaction_coord - center)**2
        
        window_data = {
            'reaction_coord': reaction_coord,
            'center': center,
            'force_constant': force_constant,
            'bias_potential': bias_potential,
            'n_samples': len(reaction_coord)
        }
        
        self.windows.append(window_data)
        logger.info(f"Added window {len(self.windows)}: center={center:.2f}, k={force_constant:.2f}, n={len(reaction_coord)}")
    
    def wham_1d(
        self,
        bins: Union[int, np.ndarray] = 50,
        tolerance: float = 1e-6,
        max_iterations: int = 10000,
        min_counts: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate PMF using 1D WHAM.
        
        Args:
            bins: Number of bins or bin edges
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            min_counts: Minimum counts per bin
            
        Returns:
            Tuple of (reaction_coordinate, pmf)
        """
        if not self.windows:
            raise ValueError("No windows added")
        
        # Collect all reaction coordinate data
        all_coords = np.concatenate([w['reaction_coord'] for w in self.windows])
        
        # Create bins
        if isinstance(bins, int):
            bin_edges = np.linspace(all_coords.min(), all_coords.max(), bins + 1)
        else:
            bin_edges = bins
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        n_bins = len(bin_centers)
        n_windows = len(self.windows)
        
        # Initialize arrays
        N_i = np.array([w['n_samples'] for w in self.windows])  # Samples per window
        n_jk = np.zeros((n_bins, n_windows))  # Histogram counts
        V_jk = np.zeros((n_bins, n_windows))  # Bias potentials
        
        # Build histograms and bias potentials
        for k, window in enumerate(self.windows):
            counts, _ = np.histogram(window['reaction_coord'], bins=bin_edges)
            n_jk[:, k] = counts
            
            # Calculate bias potential at bin centers
            V_jk[:, k] = 0.5 * window['force_constant'] * (bin_centers - window['center'])**2
        
        # WHAM iteration
        f_i = np.zeros(n_windows)  # Free energies of windows
        p_j = np.ones(n_bins) / n_bins  # Probability distribution
        
        for iteration in range(max_iterations):
            p_j_old = p_j.copy()
            
            # Update window free energies
            for i in range(n_windows):
                denominator = np.sum(p_j * np.exp(-self.beta * V_jk[:, i]))
                if denominator > 0:
                    f_i[i] = -self.kT * np.log(denominator)
            
            # Update probability distribution
            numerator = np.sum(n_jk, axis=1)
            denominator = np.zeros(n_bins)
            
            for j in range(n_bins):
                for i in range(n_windows):
                    denominator[j] += N_i[i] * np.exp(self.beta * (f_i[i] - V_jk[j, i]))
            
            # Avoid division by zero
            mask = denominator > 0
            p_j[mask] = numerator[mask] / denominator[mask]
            p_j[~mask] = 0
            
            # Normalize
            if p_j.sum() > 0:
                p_j /= p_j.sum()
            
            # Check convergence
            change = np.abs(p_j - p_j_old).max()
            if change < tolerance:
                logger.info(f"WHAM converged in {iteration + 1} iterations")
                break
        
        # Calculate PMF
        pmf = np.full(n_bins, np.nan)
        mask = p_j > 0
        pmf[mask] = -self.kT * np.log(p_j[mask])
        
        # Set minimum to zero
        pmf -= np.nanmin(pmf)
        
        # Filter bins with sufficient counts
        total_counts = np.sum(n_jk, axis=1)
        valid_mask = total_counts >= min_counts
        
        self.pmf_profile = {
            'reaction_coord': bin_centers[valid_mask],
            'pmf': pmf[valid_mask],
            'probability': p_j[valid_mask],
            'counts': total_counts[valid_mask]
        }
        
        return bin_centers[valid_mask], pmf[valid_mask]
    
    def bootstrap_error(
        self,
        n_bootstrap: int = 100,
        bins: Union[int, np.ndarray] = 50
    ) -> np.ndarray:
        """Estimate PMF error using bootstrap.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            bins: Bins for PMF calculation
            
        Returns:
            Array of PMF errors
        """
        bootstrap_pmfs = []
        
        for i in range(n_bootstrap):
            # Resample windows
            bootstrap_windows = []
            
            for window in self.windows:
                n_samples = window['n_samples']
                indices = np.random.randint(0, n_samples, size=n_samples)
                
                bootstrap_window = {
                    'reaction_coord': window['reaction_coord'][indices],
                    'center': window['center'],
                    'force_constant': window['force_constant'],
                    'bias_potential': window['bias_potential'][indices],
                    'n_samples': n_samples
                }
                bootstrap_windows.append(bootstrap_window)
            
            # Calculate PMF for bootstrap sample
            original_windows = self.windows
            self.windows = bootstrap_windows
            
            try:
                _, pmf = self.wham_1d(bins=bins)
                bootstrap_pmfs.append(pmf)
            except:
                pass
            
            self.windows = original_windows
        
        if bootstrap_pmfs:
            bootstrap_pmfs = np.array(bootstrap_pmfs)
            errors = np.std(bootstrap_pmfs, axis=0)
        else:
            errors = np.zeros(len(self.pmf_profile['pmf']))
        
        self.pmf_profile['error'] = errors
        
        return errors
    
    def plot_pmf(
        self,
        output_file: Optional[Union[str, Path]] = None,
        show_windows: bool = True,
        show_error: bool = True
    ):
        """Plot PMF profile with optional window distributions.
        
        Args:
            output_file: Optional output file path
            show_windows: Show individual window histograms
            show_error: Show error bars if available
        """
        if self.pmf_profile is None:
            raise ValueError("No PMF profile calculated")
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        
        # PMF profile
        ax1 = axes[0]
        x = self.pmf_profile['reaction_coord']
        y = self.pmf_profile['pmf']
        
        if show_error and 'error' in self.pmf_profile:
            error = self.pmf_profile['error']
            ax1.fill_between(x, y - error, y + error, alpha=0.3, color='blue')
        
        ax1.plot(x, y, 'b-', linewidth=2, label='PMF')
        ax1.set_xlabel('Reaction Coordinate (Å)')
        ax1.set_ylabel('PMF (kcal/mol)')
        ax1.set_title('Potential of Mean Force')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Window distributions
        ax2 = axes[1]
        
        if show_windows:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.windows)))
            
            for i, (window, color) in enumerate(zip(self.windows, colors)):
                hist, edges = np.histogram(window['reaction_coord'], bins=50, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                ax2.plot(centers, hist, color=color, alpha=0.6, 
                        label=f"Window {i+1} (x₀={window['center']:.1f})")
                ax2.axvline(x=window['center'], color=color, linestyle='--', alpha=0.3)
        
        ax2.set_xlabel('Reaction Coordinate (Å)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Umbrella Sampling Windows')
        ax2.legend(ncol=2, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PMF plot to {output_file}")
        else:
            plt.show()
    
    def calculate_barrier(self) -> Dict[str, float]:
        """Calculate activation barriers from PMF.
        
        Returns:
            Dictionary with barrier information
        """
        if self.pmf_profile is None:
            raise ValueError("No PMF profile calculated")
        
        x = self.pmf_profile['reaction_coord']
        y = self.pmf_profile['pmf']
        
        # Find minima and maxima
        from scipy.signal import find_peaks
        
        # Find minima (invert for find_peaks)
        minima_idx, _ = find_peaks(-y, prominence=0.5)
        
        # Find maxima
        maxima_idx, _ = find_peaks(y, prominence=0.5)
        
        results = {
            'minima_positions': x[minima_idx] if len(minima_idx) > 0 else [],
            'minima_energies': y[minima_idx] if len(minima_idx) > 0 else [],
            'maxima_positions': x[maxima_idx] if len(maxima_idx) > 0 else [],
            'maxima_energies': y[maxima_idx] if len(maxima_idx) > 0 else [],
        }
        
        # Calculate barriers
        if len(minima_idx) >= 2 and len(maxima_idx) >= 1:
            # Forward barrier (first minimum to maximum)
            forward_barrier = y[maxima_idx[0]] - y[minima_idx[0]]
            results['forward_barrier'] = forward_barrier
            
            # Reverse barrier (second minimum to maximum)
            reverse_barrier = y[maxima_idx[0]] - y[minima_idx[1]]
            results['reverse_barrier'] = reverse_barrier
            
            # Reaction free energy
            delta_g = y[minima_idx[1]] - y[minima_idx[0]]
            results['delta_g'] = delta_g
        
        return results
    
    def write_pmf(self, output_file: Union[str, Path]):
        """Write PMF profile to file.
        
        Args:
            output_file: Output file path
        """
        if self.pmf_profile is None:
            raise ValueError("No PMF profile calculated")
        
        output_file = Path(output_file)
        
        df = pd.DataFrame({
            'reaction_coord': self.pmf_profile['reaction_coord'],
            'pmf': self.pmf_profile['pmf'],
            'probability': self.pmf_profile['probability'],
            'counts': self.pmf_profile['counts']
        })
        
        if 'error' in self.pmf_profile:
            df['error'] = self.pmf_profile['error']
        
        # Write with header
        with open(output_file, 'w') as f:
            f.write(f"# PMF Analysis Results\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write(f"# Number of windows: {len(self.windows)}\n")
            f.write("#\n")
        
        df.to_csv(output_file, mode='a', index=False, float_format='%.6f')
        
        logger.info(f"Wrote PMF profile to {output_file}")
    
    def __repr__(self) -> str:
        """String representation."""
        n_windows = len(self.windows)
        return f"PMFAnalysis(T={self.temperature}K, windows={n_windows})"