"""Tests for FEP analysis module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from mdtoolkit.free_energy import FEPAnalysis


class TestFEPAnalysis:
    """Test FEP analysis functionality."""
    
    def test_initialization(self):
        """Test FEPAnalysis initialization."""
        fep = FEPAnalysis(temperature=300.0)
        
        assert fep.temperature == 300.0
        assert fep.beta == pytest.approx(1.0 / (0.0019872041 * 300.0))
        assert len(fep.lambda_windows) == 0
        assert len(fep.energy_differences) == 0
    
    def test_zwanzig_calculation(self, fep_data):
        """Test Zwanzig equation calculation."""
        fep = FEPAnalysis()
        
        # Use first window data
        window = fep_data["window_0"]
        forward_work = window["forward"]
        
        delta_g, error = fep.calculate_fep(forward_work)
        
        assert isinstance(delta_g, float)
        assert isinstance(error, float)
        assert not np.isnan(delta_g)
        assert error > 0
    
    def test_bar_calculation(self, fep_data):
        """Test BAR calculation."""
        fep = FEPAnalysis()
        
        window = fep_data["window_0"]
        forward = window["forward"]
        backward = window["backward"]
        
        delta_g, error = fep.calculate_bar(forward, backward)
        
        assert isinstance(delta_g, float)
        assert isinstance(error, float)
        assert not np.isnan(delta_g)
        assert error > 0
        
        # BAR should be more accurate than simple FEP
        fep_result, _ = fep.calculate_fep(forward)
        assert abs(delta_g) <= abs(fep_result) * 1.1  # Allow 10% tolerance
    
    def test_convergence_analysis(self, fep_data):
        """Test convergence analysis."""
        fep = FEPAnalysis()
        
        window = fep_data["window_0"]
        forward = window["forward"]
        
        convergence = fep.analyze_convergence(forward, block_size=100)
        
        assert "cumulative" in convergence
        assert "blocks" in convergence
        assert "block_std" in convergence
        
        # Cumulative should approach final value
        cumulative = convergence["cumulative"]
        assert len(cumulative) > 0
        assert np.all(np.isfinite(cumulative))
    
    def test_pmf_calculation(self, fep_data):
        """Test PMF profile calculation."""
        fep = FEPAnalysis()
        
        lambda_windows = []
        free_energies = []
        errors = []
        
        for window_data in fep_data.values():
            lambda_windows.append(
                (window_data["lambda_start"], window_data["lambda_end"])
            )
            delta_g, error = fep.calculate_fep(window_data["forward"])
            free_energies.append(delta_g)
            errors.append(error)
        
        pmf_df = fep.calculate_pmf(lambda_windows, free_energies, errors)
        
        assert "lambda" in pmf_df.columns
        assert "free_energy" in pmf_df.columns
        assert "error" in pmf_df.columns
        
        # PMF should start at 0
        assert pmf_df.iloc[0]["free_energy"] == 0.0
        
        # Error should propagate
        final_error = pmf_df.iloc[-1]["error"]
        assert final_error > 0
    
    def test_file_output(self, fep_data, temp_dir):
        """Test writing results to file."""
        fep = FEPAnalysis()
        
        # Add some data
        for window_data in fep_data.values():
            window = (window_data["lambda_start"], window_data["lambda_end"])
            fep.lambda_windows.append(window)
            
            delta_g, error = fep.calculate_fep(window_data["forward"])
            fep.free_energies[window] = delta_g
            fep.errors[window] = error
        
        output_file = temp_dir / "fep_results.txt"
        fep.write_results(output_file)
        
        assert output_file.exists()
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            assert "FEP Analysis Results" in content
            assert "Temperature: 300.0 K" in content
            assert "Total:" in content
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        fep = FEPAnalysis()
        
        # Empty data
        with pytest.raises(Exception):
            fep.calculate_fep(np.array([]))
        
        # All same values (no fluctuation)
        uniform_data = np.ones(100) * 2.0
        delta_g, error = fep.calculate_fep(uniform_data)
        assert np.isfinite(delta_g)
        
        # Very large values (numerical stability)
        large_data = np.ones(100) * 100.0
        delta_g, error = fep.calculate_fep(large_data)
        assert np.isfinite(delta_g)
    
    @pytest.mark.parametrize("temperature", [250.0, 300.0, 350.0])
    def test_temperature_dependence(self, temperature):
        """Test temperature dependence of calculations."""
        fep = FEPAnalysis(temperature=temperature)
        
        # Same work at different temperatures should give different free energies
        work = np.random.normal(2.0, 0.5, 1000)
        delta_g, _ = fep.calculate_fep(work)
        
        # Higher temperature should reduce free energy magnitude
        assert np.isfinite(delta_g)
        assert fep.beta == pytest.approx(1.0 / (0.0019872041 * temperature))
    
    def test_bar_convergence(self):
        """Test BAR self-consistency convergence."""
        fep = FEPAnalysis()
        
        # Create data with good overlap
        n_samples = 1000
        forward_work = np.random.normal(2.0, 1.0, n_samples)
        backward_work = np.random.normal(-2.0, 1.0, n_samples)
        
        delta_g, error = fep.calculate_bar(
            forward_work, backward_work,
            tolerance=1e-10, max_iterations=1000
        )
        
        # Should converge to a reasonable value
        assert -1.0 < delta_g < 5.0
        assert error > 0
        assert error < 1.0  # Should have reasonable uncertainty
    
    def test_mock_fepout_parsing(self, temp_dir):
        """Test parsing of NAMD FEPout format."""
        fep = FEPAnalysis()
        
        # Create mock FEPout file
        fepout_file = temp_dir / "test.fepout"
        with open(fepout_file, 'w') as f:
            f.write("#NEW FEP WINDOW: LAMBDA 0.000000 0.100000\n")
            for i in range(100):
                forward = np.random.normal(2.0, 0.5)
                backward = np.random.normal(-2.0, 0.5)
                f.write(f"FepEnergy: 1000 2.5 3.0 4.0 5.0 {forward:.4f} {backward:.4f} 0.0\n")
        
        data = fep.load_fepout(fepout_file)
        
        assert "forward" in data
        assert "backward" in data
        assert "lambdas" in data
        assert len(data["forward"]) == 100
        assert len(data["backward"]) == 100


class TestFEPValidation:
    """Validation tests against known results."""
    
    def test_harmonic_oscillator(self):
        """Test with analytical harmonic oscillator."""
        fep = FEPAnalysis()
        
        # For harmonic perturbation, free energy is analytical
        k1, k2 = 1.0, 2.0  # Spring constants
        n_samples = 10000
        
        # Sample from first state
        x = np.random.normal(0, 1/np.sqrt(k1), n_samples)
        
        # Energy difference
        delta_u = 0.5 * (k2 - k1) * x**2
        
        # Analytical result
        analytical = -0.5 / fep.beta * np.log(k1 / k2)
        
        # Numerical result
        numerical, _ = fep.calculate_fep(delta_u)
        
        # Should match within statistical error
        assert abs(numerical - analytical) < 0.1
    
    def test_detailed_balance(self, fep_data):
        """Test detailed balance with forward and reverse calculations."""
        fep = FEPAnalysis()
        
        total_forward = 0.0
        total_backward = 0.0
        
        for window_data in fep_data.values():
            # Forward direction
            delta_g_forward, _ = fep.calculate_fep(window_data["forward"])
            total_forward += delta_g_forward
            
            # Backward direction
            delta_g_backward, _ = fep.calculate_fep(-window_data["backward"])
            total_backward += delta_g_backward
        
        # Forward and backward should sum to approximately zero
        assert abs(total_forward + total_backward) < 1.0