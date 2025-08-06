"""Bennett Acceptance Ratio (BAR) analysis module."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BARAnalysis:
    """Bennett Acceptance Ratio analysis for free energy calculations.
    
    Implements BAR and MBAR (Multistate BAR) for optimal free energy
    estimation from simulations at multiple thermodynamic states.
    """
    
    def __init__(self, temperature: float = 300.0):
        """Initialize BAR analysis.
        
        Args:
            temperature: Temperature in Kelvin (default: 300K)
        """
        self.temperature = temperature
        self.kT = 0.0019872041 * temperature  # kT in kcal/mol
        self.beta = 1.0 / self.kT
        self.states = []
        self.free_energies = None
        self.covariance_matrix = None
        
    def add_state(
        self,
        state_id: Union[int, str],
        reduced_potentials: Dict[Union[int, str], np.ndarray],
        n_samples: Optional[int] = None
    ):
        """Add thermodynamic state data.
        
        Args:
            state_id: Identifier for this state
            reduced_potentials: Dict mapping state IDs to reduced potential arrays
                                u_kn[k,n] = U_k(x_n) / kT
            n_samples: Number of samples (inferred if not provided)
        """
        if n_samples is None:
            # Infer from first potential array
            n_samples = len(next(iter(reduced_potentials.values())))
        
        state_data = {
            'id': state_id,
            'reduced_potentials': reduced_potentials,
            'n_samples': n_samples
        }
        
        self.states.append(state_data)
        logger.info(f"Added state {state_id} with {n_samples} samples")
    
    def calculate_bar_pair(
        self,
        u_ii: np.ndarray,
        u_ij: np.ndarray,
        u_ji: np.ndarray,
        u_jj: np.ndarray,
        tolerance: float = 1e-10,
        max_iterations: int = 1000
    ) -> Tuple[float, float]:
        """Calculate free energy difference between two states using BAR.
        
        Args:
            u_ii: Reduced potential of state i evaluated at state i samples
            u_ij: Reduced potential of state j evaluated at state i samples
            u_ji: Reduced potential of state i evaluated at state j samples
            u_jj: Reduced potential of state j evaluated at state j samples
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (free_energy_difference, uncertainty)
        """
        n_i = len(u_ii)
        n_j = len(u_jj)
        
        # Work values
        w_F = u_ij - u_ii  # Forward work
        w_R = u_ji - u_jj  # Reverse work
        
        # Initial guess using exponential averaging
        df_initial = -np.log(np.mean(np.exp(-w_F))) + np.log(np.mean(np.exp(w_R)))
        df = df_initial
        
        # Self-consistent iteration
        for iteration in range(max_iterations):
            # Fermi functions
            f_F = 1.0 / (1.0 + np.exp(w_F - df))
            f_R = 1.0 / (1.0 + np.exp(-w_R - df))
            
            # Update estimate
            df_new = np.log(np.mean(f_F)) - np.log(np.mean(f_R)) + np.log(n_j / n_i)
            
            if abs(df_new - df) < tolerance:
                df = df_new
                break
            
            df = df_new
        else:
            logger.warning(f"BAR did not converge in {max_iterations} iterations")
        
        # Calculate uncertainty
        var_f_F = np.var(f_F) / n_i
        var_f_R = np.var(f_R) / n_j
        
        # Derivative terms for error propagation
        d2f_F = f_F * (1 - f_F)
        d2f_R = f_R * (1 - f_R)
        
        var_df = (var_f_F / np.mean(f_F)**2 + var_f_R / np.mean(f_R)**2 + 
                 np.mean(d2f_F**2) / (n_i * np.mean(f_F)**2) +
                 np.mean(d2f_R**2) / (n_j * np.mean(f_R)**2))
        
        uncertainty = np.sqrt(var_df)
        
        # Convert to kcal/mol
        df_kcal = df * self.kT
        uncertainty_kcal = uncertainty * self.kT
        
        return df_kcal, uncertainty_kcal
    
    def calculate_mbar(
        self,
        tolerance: float = 1e-10,
        max_iterations: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate free energies using Multistate BAR (MBAR).
        
        Returns:
            Tuple of (free_energies, uncertainties) in kcal/mol
        """
        if not self.states:
            raise ValueError("No states added")
        
        K = len(self.states)  # Number of states
        N = sum(state['n_samples'] for state in self.states)  # Total samples
        
        # Build reduced potential matrix u_kn
        u_kn = np.zeros((K, N))
        N_k = np.zeros(K, dtype=int)
        
        n_offset = 0
        for k, state in enumerate(self.states):
            n_k = state['n_samples']
            N_k[k] = n_k
            
            # Fill in reduced potentials
            for j, other_state in enumerate(self.states):
                if other_state['id'] in state['reduced_potentials']:
                    u_kn[j, n_offset:n_offset + n_k] = state['reduced_potentials'][other_state['id']]
            
            n_offset += n_k
        
        # Initialize free energies (in reduced units)
        f_k = np.zeros(K)
        
        # MBAR self-consistent iteration
        for iteration in range(max_iterations):
            f_k_old = f_k.copy()
            
            # Calculate denominator
            log_denominator = np.zeros(N)
            for n in range(N):
                max_arg = -u_kn[:, n] + f_k
                max_val = np.max(max_arg)
                log_denominator[n] = max_val + np.log(np.sum(N_k * np.exp(max_arg - max_val)))
            
            # Update free energies
            for k in range(K):
                # Find samples from state k
                n_start = sum(N_k[:k])
                n_end = n_start + N_k[k]
                
                # Calculate new estimate
                log_numerator = -np.logaddexp.reduce(u_kn[k, n_start:n_end])
                log_denominator_k = np.logaddexp.reduce(log_denominator[n_start:n_end])
                
                f_k[k] = log_numerator - log_denominator_k
            
            # Shift to set first state to zero
            f_k -= f_k[0]
            
            # Check convergence
            change = np.abs(f_k - f_k_old).max()
            if change < tolerance:
                logger.info(f"MBAR converged in {iteration + 1} iterations")
                break
        else:
            logger.warning(f"MBAR did not converge in {max_iterations} iterations")
        
        # Calculate covariance matrix
        W = np.zeros((K, N))
        for k in range(K):
            for n in range(N):
                W[k, n] = N_k[k] * np.exp(f_k[k] - u_kn[k, n]) / np.sum(N_k * np.exp(f_k - u_kn[:, n]))
        
        # Compute covariance matrix
        theta = np.eye(K) - np.ones((K, K)) / K
        W_theta = W @ W.T
        
        try:
            inverse = np.linalg.pinv(theta @ W_theta @ theta)
            covariance = theta @ inverse @ theta / self.beta**2
        except:
            logger.warning("Could not compute covariance matrix")
            covariance = np.zeros((K, K))
        
        self.free_energies = f_k * self.kT
        self.covariance_matrix = covariance
        
        uncertainties = np.sqrt(np.diag(covariance))
        
        return self.free_energies, uncertainties
    
    def calculate_pmf_mbar(
        self,
        order_parameter: np.ndarray,
        bins: Union[int, np.ndarray] = 50,
        reference_state: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate PMF along order parameter using MBAR.
        
        Args:
            order_parameter: Order parameter values for all samples
            bins: Number of bins or bin edges
            reference_state: Reference state for PMF
            
        Returns:
            Tuple of (bin_centers, pmf, uncertainty)
        """
        if self.free_energies is None:
            self.calculate_mbar()
        
        # Create bins
        if isinstance(bins, int):
            bin_edges = np.linspace(order_parameter.min(), order_parameter.max(), bins + 1)
        else:
            bin_edges = bins
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        n_bins = len(bin_centers)
        
        # Calculate PMF using MBAR weights
        pmf = np.zeros(n_bins)
        uncertainty = np.zeros(n_bins)
        
        K = len(self.states)
        N = len(order_parameter)
        
        # Build weight matrix
        u_kn = np.zeros((K, N))
        N_k = np.array([state['n_samples'] for state in self.states])
        
        n_offset = 0
        for k, state in enumerate(self.states):
            n_k = state['n_samples']
            for j, other_state in enumerate(self.states):
                if other_state['id'] in state['reduced_potentials']:
                    u_kn[j, n_offset:n_offset + n_k] = state['reduced_potentials'][other_state['id']]
            n_offset += n_k
        
        # Calculate weights for reference state
        log_weights = self.free_energies[reference_state] / self.kT - u_kn[reference_state, :]
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        
        # Calculate PMF in each bin
        for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (order_parameter >= low) & (order_parameter < high)
            if np.any(mask):
                bin_weight = np.sum(weights[mask])
                if bin_weight > 0:
                    pmf[i] = -self.kT * np.log(bin_weight)
        
        # Set minimum to zero
        pmf -= np.nanmin(pmf)
        
        return bin_centers, pmf, uncertainty
    
    def plot_free_energies(
        self,
        output_file: Optional[Union[str, Path]] = None,
        state_labels: Optional[List[str]] = None
    ):
        """Plot free energy differences.
        
        Args:
            output_file: Optional output file path
            state_labels: Optional labels for states
        """
        if self.free_energies is None:
            raise ValueError("No free energies calculated")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if state_labels is None:
            state_labels = [str(state['id']) for state in self.states]
        
        x = np.arange(len(self.free_energies))
        uncertainties = np.sqrt(np.diag(self.covariance_matrix)) if self.covariance_matrix is not None else None
        
        if uncertainties is not None:
            ax.errorbar(x, self.free_energies, yerr=uncertainties, 
                       fmt='o-', markersize=8, capsize=5, capthick=2)
        else:
            ax.plot(x, self.free_energies, 'o-', markersize=8)
        
        ax.set_xlabel('State')
        ax.set_ylabel('Free Energy (kcal/mol)')
        ax.set_title('Free Energy Profile (BAR/MBAR)')
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved free energy plot to {output_file}")
        else:
            plt.show()
    
    def write_results(self, output_file: Union[str, Path]):
        """Write BAR/MBAR results to file.
        
        Args:
            output_file: Output file path
        """
        if self.free_energies is None:
            raise ValueError("No free energies calculated")
        
        output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("# BAR/MBAR Analysis Results\n")
            f.write(f"# Temperature: {self.temperature} K\n")
            f.write(f"# Number of states: {len(self.states)}\n")
            f.write("#\n")
            f.write("# State Free_Energy(kcal/mol) Uncertainty(kcal/mol) N_samples\n")
            
            uncertainties = np.sqrt(np.diag(self.covariance_matrix)) if self.covariance_matrix is not None else np.zeros_like(self.free_energies)
            
            for i, state in enumerate(self.states):
                f.write(f"{state['id']} {self.free_energies[i]:.6f} {uncertainties[i]:.6f} {state['n_samples']}\n")
            
            if len(self.states) > 1:
                f.write("#\n# Pairwise free energy differences:\n")
                for i in range(len(self.states) - 1):
                    df = self.free_energies[i+1] - self.free_energies[i]
                    f.write(f"# {self.states[i]['id']} -> {self.states[i+1]['id']}: {df:.4f} kcal/mol\n")
        
        logger.info(f"Wrote BAR/MBAR results to {output_file}")
    
    def __repr__(self) -> str:
        """String representation."""
        n_states = len(self.states)
        if self.free_energies is not None:
            return f"BARAnalysis(T={self.temperature}K, states={n_states}, converged=True)"
        else:
            return f"BARAnalysis(T={self.temperature}K, states={n_states}, converged=False)"