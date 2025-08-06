"""Free Energy Perturbation (FEP) analysis module."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FEPAnalysis:
    """Free Energy Perturbation analysis for alchemical transformations.
    
    Implements FEP analysis using the Zwanzig equation and BAR estimator
    for calculating free energy differences from molecular dynamics simulations.
    """
    
    def __init__(self, temperature: float = 300.0):
        """Initialize FEP analysis.
        
        Args:
            temperature: Temperature in Kelvin (default: 300K)
        """
        self.temperature = temperature
        self.beta = 1.0 / (0.0019872041 * temperature)  # 1/kT in kcal/mol
        self.lambda_windows = []
        self.energy_differences = {}
        self.free_energies = {}
        self.errors = {}
        
    def load_fepout(self, fepout_file: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load NAMD FEPout file.
        
        Args:
            fepout_file: Path to FEPout file
            
        Returns:
            Dictionary with forward and backward energy differences
        """
        fepout_file = Path(fepout_file)
        
        forward_dE = []
        backward_dE = []
        lambda_values = []
        
        with open(fepout_file, 'r') as f:
            for line in f:
                if line.startswith('#NEW FEP WINDOW'):
                    parts = line.split()
                    lambda1 = float(parts[6])
                    lambda2 = float(parts[8])
                    lambda_values.append((lambda1, lambda2))
                    
                elif line.startswith('FepEnergy:'):
                    parts = line.split()
                    if len(parts) >= 8:
                        forward = float(parts[6])
                        backward = float(parts[7])
                        forward_dE.append(forward)
                        backward_dE.append(backward)
        
        logger.info(f"Loaded {len(forward_dE)} energy samples from {fepout_file}")
        
        return {
            'forward': np.array(forward_dE),
            'backward': np.array(backward_dE),
            'lambdas': lambda_values
        }
    
    def calculate_fep(self, energy_differences: np.ndarray) -> Tuple[float, float]:
        """Calculate free energy using Zwanzig equation.
        
        Args:
            energy_differences: Array of energy differences (kcal/mol)
            
        Returns:
            Tuple of (free_energy, error) in kcal/mol
        """
        # Zwanzig equation: ΔG = -kT * ln(<exp(-βΔE)>)
        exp_avg = np.mean(np.exp(-self.beta * energy_differences))
        
        if exp_avg <= 0:
            logger.warning("Invalid exponential average, returning NaN")
            return np.nan, np.nan
            
        delta_g = -1.0 / self.beta * np.log(exp_avg)
        
        # Bootstrap error estimation
        n_bootstrap = 1000
        bootstrap_results = []
        n_samples = len(energy_differences)
        
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_sample = energy_differences[indices]
            exp_avg_boot = np.mean(np.exp(-self.beta * bootstrap_sample))
            if exp_avg_boot > 0:
                delta_g_boot = -1.0 / self.beta * np.log(exp_avg_boot)
                bootstrap_results.append(delta_g_boot)
        
        error = np.std(bootstrap_results) if bootstrap_results else 0.0
        
        return delta_g, error
    
    def calculate_bar(
        self,
        forward_work: np.ndarray,
        backward_work: np.ndarray,
        tolerance: float = 1e-6,
        max_iterations: int = 1000
    ) -> Tuple[float, float]:
        """Calculate free energy using Bennett Acceptance Ratio (BAR).
        
        Args:
            forward_work: Forward work values
            backward_work: Backward work values
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (free_energy, error) in kcal/mol
        """
        n_forward = len(forward_work)
        n_backward = len(backward_work)
        
        # Initial guess using FEP
        delta_f = -1.0 / self.beta * np.log(n_backward / n_forward)
        
        # Iterate to convergence
        for iteration in range(max_iterations):
            # Forward average
            f_term = 1.0 / (1.0 + np.exp(self.beta * (forward_work - delta_f)))
            forward_avg = np.mean(f_term)
            
            # Backward average
            b_term = 1.0 / (1.0 + np.exp(-self.beta * (backward_work + delta_f)))
            backward_avg = np.mean(b_term)
            
            # Update delta_f
            delta_f_new = (1.0 / self.beta) * np.log(forward_avg / backward_avg) + \
                         (1.0 / self.beta) * np.log(n_backward / n_forward)
            
            if abs(delta_f_new - delta_f) < tolerance:
                break
                
            delta_f = delta_f_new
        
        # Error estimation using analytical formula
        var_forward = np.var(f_term) / n_forward
        var_backward = np.var(b_term) / n_backward
        error = np.sqrt(var_forward + var_backward) / self.beta
        
        return delta_f, error
    
    def analyze_convergence(
        self,
        energy_differences: np.ndarray,
        block_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """Analyze convergence of free energy calculation.
        
        Args:
            energy_differences: Energy difference array
            block_size: Size of blocks for analysis
            
        Returns:
            Dictionary with convergence metrics
        """
        n_samples = len(energy_differences)
        n_blocks = n_samples // block_size
        
        cumulative_fe = []
        block_averages = []
        
        # Cumulative free energy
        for i in range(block_size, n_samples, block_size):
            fe, _ = self.calculate_fep(energy_differences[:i])
            cumulative_fe.append(fe)
        
        # Block averages
        for i in range(n_blocks):
            block = energy_differences[i*block_size:(i+1)*block_size]
            fe, _ = self.calculate_fep(block)
            block_averages.append(fe)
        
        return {
            'cumulative': np.array(cumulative_fe),
            'blocks': np.array(block_averages),
            'block_std': np.std(block_averages)
        }
    
    def plot_convergence(
        self,
        convergence_data: Dict[str, np.ndarray],
        output_file: Optional[Union[str, Path]] = None
    ):
        """Plot convergence analysis.
        
        Args:
            convergence_data: Output from analyze_convergence
            output_file: Optional output file path
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Cumulative free energy
        ax1.plot(convergence_data['cumulative'], 'b-', linewidth=2)
        ax1.axhline(y=convergence_data['cumulative'][-1], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Sample blocks')
        ax1.set_ylabel('Cumulative ΔG (kcal/mol)')
        ax1.set_title('Free Energy Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Block averages
        ax2.plot(convergence_data['blocks'], 'go-', alpha=0.7)
        mean_fe = np.mean(convergence_data['blocks'])
        std_fe = convergence_data['block_std']
        ax2.axhline(y=mean_fe, color='r', linestyle='-', label=f'Mean: {mean_fe:.2f}')
        ax2.fill_between(
            range(len(convergence_data['blocks'])),
            mean_fe - std_fe,
            mean_fe + std_fe,
            alpha=0.3,
            color='red',
            label=f'±σ: {std_fe:.2f}'
        )
        ax2.set_xlabel('Block number')
        ax2.set_ylabel('Block ΔG (kcal/mol)')
        ax2.set_title('Block Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence plot to {output_file}")
        else:
            plt.show()
    
    def calculate_pmf(
        self,
        lambda_windows: List[Tuple[float, float]],
        free_energies: List[float],
        errors: List[float]
    ) -> pd.DataFrame:
        """Calculate potential of mean force from lambda windows.
        
        Args:
            lambda_windows: List of (lambda_start, lambda_end) tuples
            free_energies: Free energy for each window
            errors: Error estimates for each window
            
        Returns:
            DataFrame with PMF profile
        """
        # Build cumulative free energy profile
        lambda_values = [0.0]
        cumulative_fe = [0.0]
        cumulative_error = [0.0]
        
        for i, (window, fe, err) in enumerate(zip(lambda_windows, free_energies, errors)):
            lambda_values.append(window[1])
            cumulative_fe.append(cumulative_fe[-1] + fe)
            cumulative_error.append(np.sqrt(cumulative_error[-1]**2 + err**2))
        
        pmf_df = pd.DataFrame({
            'lambda': lambda_values,
            'free_energy': cumulative_fe,
            'error': cumulative_error
        })
        
        return pmf_df
    
    def write_results(self, output_file: Union[str, Path]):
        """Write FEP analysis results to file.
        
        Args:
            output_file: Output file path
        """
        output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("# FEP Analysis Results\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write("#\n")
            f.write("# Lambda_start Lambda_end DeltaG(kcal/mol) Error(kcal/mol)\n")
            
            total_fe = 0.0
            total_error_sq = 0.0
            
            for window in self.lambda_windows:
                if window in self.free_energies:
                    fe = self.free_energies[window]
                    err = self.errors.get(window, 0.0)
                    f.write(f"{window[0]:.3f} {window[1]:.3f} {fe:.4f} {err:.4f}\n")
                    total_fe += fe
                    total_error_sq += err**2
            
            total_error = np.sqrt(total_error_sq)
            f.write(f"#\n# Total: {total_fe:.4f} +/- {total_error:.4f} kcal/mol\n")
        
        logger.info(f"Wrote results to {output_file}")
    
    def __repr__(self) -> str:
        """String representation."""
        n_windows = len(self.lambda_windows)
        return f"FEPAnalysis(T={self.temperature}K, windows={n_windows})"