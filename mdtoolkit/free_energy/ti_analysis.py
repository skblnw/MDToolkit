"""Thermodynamic Integration (TI) analysis module."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TIAnalysis:
    """Thermodynamic Integration analysis for free energy calculations.
    
    Implements TI analysis using numerical integration of dU/dλ
    for calculating free energy differences along alchemical pathways.
    """
    
    def __init__(self, temperature: float = 300.0):
        """Initialize TI analysis.
        
        Args:
            temperature: Temperature in Kelvin (default: 300K)
        """
        self.temperature = temperature
        self.beta = 1.0 / (0.0019872041 * temperature)  # 1/kT in kcal/mol
        self.lambda_values = []
        self.dudl_values = {}
        self.free_energy = None
        self.error = None
        
    def add_lambda_window(
        self,
        lambda_value: float,
        dudl_data: np.ndarray,
        weights: Optional[np.ndarray] = None
    ):
        """Add dU/dλ data for a lambda window.
        
        Args:
            lambda_value: Lambda value for this window
            dudl_data: Array of dU/dλ values (kcal/mol)
            weights: Optional weights for weighted averaging
        """
        if lambda_value not in self.lambda_values:
            self.lambda_values.append(lambda_value)
            
        if weights is None:
            weights = np.ones_like(dudl_data)
            
        # Calculate weighted average and error
        mean_dudl = np.average(dudl_data, weights=weights)
        
        # Bootstrap error estimation
        n_bootstrap = 1000
        bootstrap_means = []
        n_samples = len(dudl_data)
        
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_sample = dudl_data[indices]
            bootstrap_weights = weights[indices]
            bootstrap_mean = np.average(bootstrap_sample, weights=bootstrap_weights)
            bootstrap_means.append(bootstrap_mean)
        
        error_dudl = np.std(bootstrap_means)
        
        self.dudl_values[lambda_value] = {
            'mean': mean_dudl,
            'error': error_dudl,
            'raw_data': dudl_data,
            'weights': weights
        }
        
        logger.info(f"Added λ={lambda_value:.3f}: <dU/dλ>={mean_dudl:.4f} ± {error_dudl:.4f}")
    
    def integrate_ti(
        self,
        method: str = 'trapezoid',
        spline_order: int = 3
    ) -> Tuple[float, float]:
        """Integrate dU/dλ to calculate free energy.
        
        Args:
            method: Integration method ('trapezoid', 'simpson', 'spline')
            spline_order: Order of spline for spline integration
            
        Returns:
            Tuple of (free_energy, error) in kcal/mol
        """
        if not self.lambda_values:
            raise ValueError("No lambda windows added")
        
        # Sort lambda values
        sorted_lambdas = sorted(self.lambda_values)
        mean_dudl = np.array([self.dudl_values[lam]['mean'] for lam in sorted_lambdas])
        errors = np.array([self.dudl_values[lam]['error'] for lam in sorted_lambdas])
        
        if method == 'trapezoid':
            # Trapezoidal integration
            self.free_energy = integrate.trapezoid(mean_dudl, sorted_lambdas)
            
            # Error propagation for trapezoidal rule
            weights = np.diff(sorted_lambdas)
            weighted_errors = np.zeros(len(sorted_lambdas))
            weighted_errors[:-1] += 0.5 * weights * errors[:-1]
            weighted_errors[1:] += 0.5 * weights * errors[1:]
            self.error = np.sqrt(np.sum(weighted_errors**2))
            
        elif method == 'simpson':
            # Simpson's rule integration
            if len(sorted_lambdas) % 2 == 0:
                logger.warning("Simpson's rule requires odd number of points, using trapezoidal")
                return self.integrate_ti(method='trapezoid')
                
            self.free_energy = integrate.simpson(mean_dudl, sorted_lambdas)
            
            # Simplified error propagation
            self.error = np.sqrt(np.sum(errors**2)) * (sorted_lambdas[-1] - sorted_lambdas[0]) / len(sorted_lambdas)
            
        elif method == 'spline':
            # Spline interpolation and integration
            spline = interpolate.UnivariateSpline(
                sorted_lambdas, mean_dudl, 
                k=min(spline_order, len(sorted_lambdas)-1),
                s=0
            )
            self.free_energy = spline.integral(sorted_lambdas[0], sorted_lambdas[-1])
            
            # Error estimation using bootstrap on spline parameters
            n_bootstrap = 1000
            bootstrap_integrals = []
            
            for _ in range(n_bootstrap):
                # Perturb dU/dλ values within error bounds
                perturbed_dudl = mean_dudl + np.random.normal(0, errors)
                spline_boot = interpolate.UnivariateSpline(
                    sorted_lambdas, perturbed_dudl,
                    k=min(spline_order, len(sorted_lambdas)-1),
                    s=0
                )
                integral = spline_boot.integral(sorted_lambdas[0], sorted_lambdas[-1])
                bootstrap_integrals.append(integral)
            
            self.error = np.std(bootstrap_integrals)
        
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        logger.info(f"TI integration ({method}): ΔG = {self.free_energy:.4f} ± {self.error:.4f} kcal/mol")
        
        return self.free_energy, self.error
    
    def plot_ti_curve(
        self,
        output_file: Optional[Union[str, Path]] = None,
        show_spline: bool = True,
        show_data: bool = True
    ):
        """Plot TI integration curve.
        
        Args:
            output_file: Optional output file path
            show_spline: Show spline interpolation
            show_data: Show raw data points
        """
        sorted_lambdas = sorted(self.lambda_values)
        mean_dudl = np.array([self.dudl_values[lam]['mean'] for lam in sorted_lambdas])
        errors = np.array([self.dudl_values[lam]['error'] for lam in sorted_lambdas])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # dU/dλ vs λ
        if show_data:
            ax1.errorbar(
                sorted_lambdas, mean_dudl, yerr=errors,
                fmt='o', markersize=8, capsize=5, capthick=2,
                label='Data', color='blue'
            )
        
        if show_spline and len(sorted_lambdas) > 3:
            # Fit spline for smooth curve
            lambda_smooth = np.linspace(sorted_lambdas[0], sorted_lambdas[-1], 1000)
            spline = interpolate.UnivariateSpline(sorted_lambdas, mean_dudl, k=3, s=0)
            dudl_smooth = spline(lambda_smooth)
            ax1.plot(lambda_smooth, dudl_smooth, 'r-', alpha=0.7, label='Spline fit')
        
        ax1.set_xlabel('λ')
        ax1.set_ylabel('⟨∂U/∂λ⟩ (kcal/mol)')
        ax1.set_title('Thermodynamic Integration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative integral
        cumulative_integral = []
        lambda_points = []
        
        for i in range(1, len(sorted_lambdas)):
            sub_lambdas = sorted_lambdas[:i+1]
            sub_dudl = mean_dudl[:i+1]
            integral = integrate.trapezoid(sub_dudl, sub_lambdas)
            cumulative_integral.append(integral)
            lambda_points.append(sorted_lambdas[i])
        
        ax2.plot(lambda_points, cumulative_integral, 'g-', linewidth=2)
        ax2.axhline(y=self.free_energy, color='r', linestyle='--', alpha=0.7,
                   label=f'ΔG = {self.free_energy:.3f} ± {self.error:.3f} kcal/mol')
        ax2.set_xlabel('λ')
        ax2.set_ylabel('Cumulative ΔG (kcal/mol)')
        ax2.set_title('Cumulative Free Energy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved TI plot to {output_file}")
        else:
            plt.show()
    
    def analyze_overlap(self) -> Dict[str, float]:
        """Analyze phase space overlap between adjacent lambda windows.
        
        Returns:
            Dictionary with overlap metrics
        """
        sorted_lambdas = sorted(self.lambda_values)
        
        if len(sorted_lambdas) < 2:
            return {}
        
        overlaps = []
        
        for i in range(len(sorted_lambdas) - 1):
            lambda1 = sorted_lambdas[i]
            lambda2 = sorted_lambdas[i + 1]
            
            # Get raw dU/dλ distributions
            dist1 = self.dudl_values[lambda1]['raw_data']
            dist2 = self.dudl_values[lambda2]['raw_data']
            
            # Calculate overlap using histogram method
            min_val = min(dist1.min(), dist2.min())
            max_val = max(dist1.max(), dist2.max())
            bins = np.linspace(min_val, max_val, 50)
            
            hist1, _ = np.histogram(dist1, bins=bins, density=True)
            hist2, _ = np.histogram(dist2, bins=bins, density=True)
            
            # Overlap coefficient
            overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
            overlaps.append(overlap)
            
            logger.info(f"Overlap between λ={lambda1:.3f} and λ={lambda2:.3f}: {overlap:.3f}")
        
        return {
            'overlaps': overlaps,
            'mean_overlap': np.mean(overlaps),
            'min_overlap': np.min(overlaps),
            'lambda_pairs': list(zip(sorted_lambdas[:-1], sorted_lambdas[1:]))
        }
    
    def write_results(self, output_file: Union[str, Path]):
        """Write TI analysis results to file.
        
        Args:
            output_file: Output file path
        """
        output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("# Thermodynamic Integration Analysis Results\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write("#\n")
            f.write("# Lambda <dU/dλ>(kcal/mol) Error(kcal/mol)\n")
            
            sorted_lambdas = sorted(self.lambda_values)
            for lam in sorted_lambdas:
                mean = self.dudl_values[lam]['mean']
                error = self.dudl_values[lam]['error']
                f.write(f"{lam:.4f} {mean:.6f} {error:.6f}\n")
            
            if self.free_energy is not None:
                f.write(f"#\n# Free Energy: {self.free_energy:.4f} ± {self.error:.4f} kcal/mol\n")
        
        logger.info(f"Wrote TI results to {output_file}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with lambda, dU/dλ, and errors
        """
        sorted_lambdas = sorted(self.lambda_values)
        
        data = {
            'lambda': sorted_lambdas,
            'dudl_mean': [self.dudl_values[lam]['mean'] for lam in sorted_lambdas],
            'dudl_error': [self.dudl_values[lam]['error'] for lam in sorted_lambdas],
            'n_samples': [len(self.dudl_values[lam]['raw_data']) for lam in sorted_lambdas]
        }
        
        df = pd.DataFrame(data)
        
        if self.free_energy is not None:
            df.attrs['free_energy'] = self.free_energy
            df.attrs['error'] = self.error
        
        return df
    
    def __repr__(self) -> str:
        """String representation."""
        n_windows = len(self.lambda_values)
        if self.free_energy is not None:
            return f"TIAnalysis(T={self.temperature}K, windows={n_windows}, ΔG={self.free_energy:.3f}±{self.error:.3f})"
        else:
            return f"TIAnalysis(T={self.temperature}K, windows={n_windows})"