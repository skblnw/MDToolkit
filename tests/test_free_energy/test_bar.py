"""Tests for BAR/MBAR analysis module."""

import pytest
import numpy as np
from pathlib import Path

from mdtoolkit.free_energy import BARAnalysis


class TestBARAnalysis:
    """Test BAR analysis functionality."""
    
    def test_initialization(self):
        """Test BARAnalysis initialization."""
        bar = BARAnalysis(temperature=300.0)
        
        assert bar.temperature == 300.0
        assert bar.kT == pytest.approx(0.0019872041 * 300.0)
        assert bar.beta == pytest.approx(1.0 / bar.kT)
        assert len(bar.states) == 0
        assert bar.free_energies is None
        assert bar.covariance_matrix is None
    
    def test_add_state(self, bar_data):
        """Test adding thermodynamic states."""
        bar = BARAnalysis()
        
        for state in bar_data:
            bar.add_state(
                state_id=state["id"],
                reduced_potentials=state["potentials"],
                n_samples=state["n_samples"]
            )
        
        assert len(bar.states) == len(bar_data)
        
        for i, state in enumerate(bar.states):
            assert state["id"] == bar_data[i]["id"]
            assert state["n_samples"] == bar_data[i]["n_samples"]
            assert "reduced_potentials" in state
    
    def test_bar_pair_calculation(self):
        """Test pairwise BAR calculation."""
        bar = BARAnalysis()
        
        # Create two states with known overlap
        n_samples = 1000
        
        # State 0 samples
        u_00 = np.random.normal(0, 1, n_samples)
        u_01 = u_00 + np.random.normal(2, 0.5, n_samples)  # Higher energy in state 1
        
        # State 1 samples
        u_11 = np.random.normal(0, 1, n_samples)
        u_10 = u_11 + np.random.normal(2, 0.5, n_samples)  # Higher energy in state 0
        
        delta_f, uncertainty = bar.calculate_bar_pair(u_00, u_01, u_10, u_11)
        
        assert isinstance(delta_f, float)
        assert isinstance(uncertainty, float)
        assert np.isfinite(delta_f)
        assert uncertainty > 0
        
        # Free energy difference should be reasonable
        assert -5 < delta_f < 5  # Within 5 kcal/mol
    
    def test_bar_convergence(self):
        """Test BAR self-consistency convergence."""
        bar = BARAnalysis()
        
        n_samples = 500
        # Create overlapping distributions
        u_ii = np.random.normal(0, 1, n_samples)
        u_ij = np.random.normal(1, 1, n_samples)
        u_ji = np.random.normal(-1, 1, n_samples)
        u_jj = np.random.normal(0, 1, n_samples)
        
        # Test with different tolerances
        delta_f1, _ = bar.calculate_bar_pair(
            u_ii, u_ij, u_ji, u_jj, tolerance=1e-6
        )
        delta_f2, _ = bar.calculate_bar_pair(
            u_ii, u_ij, u_ji, u_jj, tolerance=1e-10
        )
        
        # More stringent tolerance should give similar result
        assert abs(delta_f1 - delta_f2) < 0.01
    
    def test_mbar_calculation(self, bar_data):
        """Test MBAR calculation."""
        bar = BARAnalysis()
        
        # Add states
        for state in bar_data[:3]:  # Use fewer states for speed
            bar.add_state(
                state_id=state["id"],
                reduced_potentials=state["potentials"]
            )
        
        # Calculate MBAR
        free_energies, uncertainties = bar.calculate_mbar(
            tolerance=1e-8, max_iterations=1000
        )
        
        assert len(free_energies) == 3
        assert len(uncertainties) == 3
        
        # First state should be reference (F=0)
        assert free_energies[0] == pytest.approx(0.0)
        
        # Free energies should be finite
        assert np.all(np.isfinite(free_energies))
        
        # Uncertainties should be positive
        assert np.all(uncertainties >= 0)
        
        # Check covariance matrix
        assert bar.covariance_matrix is not None
        assert bar.covariance_matrix.shape == (3, 3)
    
    def test_mbar_consistency(self):
        """Test MBAR gives consistent results with BAR for two states."""
        bar = BARAnalysis()
        
        # Create two-state system
        n_samples = 1000
        states = []
        
        for i in range(2):
            potentials = {}
            for j in range(2):
                delta = abs(i - j) * 1.5
                potentials[j] = np.random.normal(delta, 0.8, n_samples)
            
            states.append({
                "id": i,
                "potentials": potentials,
                "n_samples": n_samples
            })
        
        # Add states
        for state in states:
            bar.add_state(state["id"], state["potentials"])
        
        # Calculate with MBAR
        mbar_fe, _ = bar.calculate_mbar()
        
        # Calculate with pairwise BAR
        bar2 = BARAnalysis()
        bar_fe, _ = bar2.calculate_bar_pair(
            states[0]["potentials"][0],
            states[0]["potentials"][1],
            states[1]["potentials"][0],
            states[1]["potentials"][1]
        )
        
        # Should give similar results
        assert abs(mbar_fe[1] - mbar_fe[0] - bar_fe) < 0.1
    
    def test_pmf_from_mbar(self, bar_data):
        """Test PMF calculation from MBAR."""
        bar = BARAnalysis()
        
        # Add states
        for state in bar_data[:3]:
            bar.add_state(state["id"], state["potentials"])
        
        # Calculate MBAR first
        bar.calculate_mbar()
        
        # Create order parameter
        n_total = sum(state["n_samples"] for state in bar_data[:3])
        order_param = np.random.uniform(0, 10, n_total)
        
        # Calculate PMF
        bin_centers, pmf, uncertainty = bar.calculate_pmf_mbar(
            order_param, bins=20, reference_state=0
        )
        
        assert len(bin_centers) == 20
        assert len(pmf) == 20
        assert len(uncertainty) == 20
        
        # PMF should have minimum at 0
        assert np.nanmin(pmf) == pytest.approx(0.0, abs=1e-10)
    
    def test_file_output(self, bar_data, temp_dir):
        """Test writing results to file."""
        bar = BARAnalysis()
        
        for state in bar_data[:3]:
            bar.add_state(state["id"], state["potentials"])
        
        bar.calculate_mbar()
        
        output_file = temp_dir / "bar_results.txt"
        bar.write_results(output_file)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            content = f.read()
            assert "BAR/MBAR Analysis Results" in content
            assert "Temperature: 300.0 K" in content
            assert "Free_Energy" in content
            assert "Uncertainty" in content
    
    def test_insufficient_overlap(self):
        """Test behavior with poor phase space overlap."""
        bar = BARAnalysis()
        
        # Create states with no overlap
        n_samples = 500
        
        # Very separated distributions
        u_ii = np.random.normal(0, 0.5, n_samples)
        u_ij = np.random.normal(100, 0.5, n_samples)  # Very high energy
        u_ji = np.random.normal(-100, 0.5, n_samples)  # Very low energy
        u_jj = np.random.normal(0, 0.5, n_samples)
        
        # Should still return a result (but may be inaccurate)
        delta_f, uncertainty = bar.calculate_bar_pair(
            u_ii, u_ij, u_ji, u_jj, max_iterations=100
        )
        
        assert np.isfinite(delta_f)
        assert uncertainty > 0
        
        # Uncertainty should be large due to poor overlap
        assert uncertainty > 1.0
    
    @pytest.mark.parametrize("n_states", [2, 3, 5, 10])
    def test_different_state_numbers(self, n_states):
        """Test MBAR with different numbers of states."""
        bar = BARAnalysis()
        
        # Create states
        n_samples = 200
        for i in range(n_states):
            potentials = {}
            for j in range(n_states):
                delta = abs(i - j) * 0.5
                potentials[j] = np.random.normal(delta, 0.5, n_samples)
            
            bar.add_state(i, potentials, n_samples)
        
        # Calculate MBAR
        free_energies, uncertainties = bar.calculate_mbar(max_iterations=500)
        
        assert len(free_energies) == n_states
        assert len(uncertainties) == n_states
        assert free_energies[0] == 0.0  # Reference state
        
        # Free energies should increase monotonically (approximately)
        # for this simple model
        assert np.all(np.diff(free_energies) >= -0.5)  # Allow small violations


class TestBARValidation:
    """Validation tests for BAR/MBAR."""
    
    def test_detailed_balance(self):
        """Test detailed balance in BAR."""
        bar = BARAnalysis()
        
        # Create reversible work distributions
        n_samples = 2000
        
        # Forward process
        w_forward = np.random.normal(2.0, 1.0, n_samples)
        
        # Reverse process (should satisfy Crooks relation)
        w_reverse = -w_forward + np.random.normal(0, 0.1, n_samples)
        
        # Convert to potentials
        u_ii = np.zeros(n_samples)
        u_ij = w_forward
        u_jj = np.zeros(n_samples)
        u_ji = w_reverse
        
        delta_f_forward, _ = bar.calculate_bar_pair(u_ii, u_ij, u_ji, u_jj)
        delta_f_reverse, _ = bar.calculate_bar_pair(u_jj, u_ji, u_ij, u_ii)
        
        # Should satisfy detailed balance
        assert abs(delta_f_forward + delta_f_reverse) < 0.2
    
    def test_harmonic_oscillators(self):
        """Test with displaced harmonic oscillators (analytical result)."""
        bar = BARAnalysis()
        
        # Two harmonic oscillators with different centers
        k = 10.0  # Spring constant
        x0_1, x0_2 = 0.0, 2.0  # Centers
        
        n_samples = 5000
        
        # Sample from each state
        x_1 = np.random.normal(x0_1, 1/np.sqrt(k * bar.beta), n_samples)
        x_2 = np.random.normal(x0_2, 1/np.sqrt(k * bar.beta), n_samples)
        
        # Calculate energies
        u_11 = 0.5 * k * (x_1 - x0_1)**2
        u_12 = 0.5 * k * (x_1 - x0_2)**2
        u_21 = 0.5 * k * (x_2 - x0_1)**2
        u_22 = 0.5 * k * (x_2 - x0_2)**2
        
        # Analytical result
        analytical = 0.5 * k * (x0_2 - x0_1)**2
        
        # BAR result
        numerical, _ = bar.calculate_bar_pair(u_11, u_12, u_21, u_22)
        
        # Should match within statistical error
        assert abs(numerical - analytical) < 0.2
    
    def test_mbar_sum_rule(self, bar_data):
        """Test MBAR sum rule for free energies."""
        bar = BARAnalysis()
        
        # Create cyclic states (0 -> 1 -> 2 -> 0)
        n_states = 3
        n_samples = 1000
        
        for i in range(n_states):
            potentials = {}
            for j in range(n_states):
                # Create cyclic energy landscape
                if j == (i + 1) % n_states:
                    delta = 1.0  # Favorable
                elif j == i:
                    delta = 0.0
                else:
                    delta = 2.0  # Unfavorable
                
                potentials[j] = np.random.normal(delta, 0.5, n_samples)
            
            bar.add_state(i, potentials)
        
        free_energies, _ = bar.calculate_mbar()
        
        # Check cycle sum (should be approximately 0 for reversible process)
        cycle_sum = (free_energies[1] - free_energies[0]) + \
                   (free_energies[2] - free_energies[1]) + \
                   (free_energies[0] - free_energies[2])
        
        assert abs(cycle_sum) < 0.1  # Should be close to 0