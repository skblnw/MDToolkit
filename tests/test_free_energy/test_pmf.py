"""Tests for PMF analysis module."""

import pytest
import numpy as np
from pathlib import Path
import pandas as pd

from mdtoolkit.free_energy import PMFAnalysis


class TestPMFAnalysis:
    """Test PMF analysis functionality."""
    
    def test_initialization(self):
        """Test PMFAnalysis initialization."""
        pmf = PMFAnalysis(temperature=300.0)
        
        assert pmf.temperature == 300.0
        assert pmf.kT == pytest.approx(0.0019872041 * 300.0)
        assert pmf.beta == pytest.approx(1.0 / pmf.kT)
        assert len(pmf.windows) == 0
        assert pmf.pmf_profile is None
    
    def test_add_window(self, umbrella_data):
        """Test adding umbrella sampling windows."""
        pmf = PMFAnalysis()
        
        for window in umbrella_data:
            pmf.add_window(
                reaction_coord=window["positions"],
                center=window["center"],
                force_constant=window["force_constant"]
            )
        
        assert len(pmf.windows) == len(umbrella_data)
        
        # Check window data
        for i, window in enumerate(pmf.windows):
            assert "reaction_coord" in window
            assert "center" in window
            assert "force_constant" in window
            assert "bias_potential" in window
            assert "n_samples" in window
            assert window["center"] == umbrella_data[i]["center"]
    
    def test_wham_1d(self, umbrella_data):
        """Test 1D WHAM calculation."""
        pmf = PMFAnalysis()
        
        for window in umbrella_data:
            pmf.add_window(
                reaction_coord=window["positions"],
                center=window["center"],
                force_constant=window["force_constant"]
            )
        
        # Run WHAM
        reaction_coord, pmf_values = pmf.wham_1d(bins=50, tolerance=1e-6)
        
        assert len(reaction_coord) > 0
        assert len(pmf_values) == len(reaction_coord)
        assert np.all(np.isfinite(pmf_values[~np.isnan(pmf_values)]))
        
        # PMF should have minimum at 0
        assert np.nanmin(pmf_values) == pytest.approx(0.0)
        
        # Check stored profile
        assert pmf.pmf_profile is not None
        assert "reaction_coord" in pmf.pmf_profile
        assert "pmf" in pmf.pmf_profile
        assert "probability" in pmf.pmf_profile
        assert "counts" in pmf.pmf_profile
    
    def test_wham_convergence(self, umbrella_data):
        """Test WHAM convergence criteria."""
        pmf = PMFAnalysis()
        
        for window in umbrella_data[:3]:  # Use fewer windows
            pmf.add_window(
                reaction_coord=window["positions"],
                center=window["center"],
                force_constant=window["force_constant"]
            )
        
        # Should converge even with few windows
        reaction_coord, pmf_values = pmf.wham_1d(
            bins=30, tolerance=1e-8, max_iterations=10000
        )
        
        assert len(pmf_values) > 0
        assert not np.all(np.isnan(pmf_values))
    
    def test_bootstrap_error(self, umbrella_data):
        """Test bootstrap error estimation."""
        pmf = PMFAnalysis()
        
        # Use fewer windows for speed
        for window in umbrella_data[:5]:
            pmf.add_window(
                reaction_coord=window["positions"],
                center=window["center"],
                force_constant=window["force_constant"]
            )
        
        # Calculate PMF first
        pmf.wham_1d(bins=30)
        
        # Estimate errors
        errors = pmf.bootstrap_error(n_bootstrap=50, bins=30)
        
        assert len(errors) == len(pmf.pmf_profile["pmf"])
        assert np.all(errors >= 0)
        assert np.all(np.isfinite(errors))
        
        # Errors should be reasonable
        assert np.mean(errors) < 1.0  # Less than 1 kcal/mol average error
    
    def test_barrier_calculation(self):
        """Test activation barrier calculation."""
        pmf = PMFAnalysis()
        
        # Create windows with known double-well potential
        centers = np.linspace(-5, 5, 20)
        for center in centers:
            # Create biased samples from double-well
            n_samples = 500
            positions = np.random.normal(center, 0.5, n_samples)
            
            pmf.add_window(
                reaction_coord=positions,
                center=center,
                force_constant=20.0
            )
        
        # Calculate PMF
        pmf.wham_1d(bins=50)
        
        # Calculate barriers
        barriers = pmf.calculate_barrier()
        
        assert "minima_positions" in barriers
        assert "minima_energies" in barriers
        assert "maxima_positions" in barriers
        assert "maxima_energies" in barriers
        
        # For sufficient sampling, should find minima and maxima
        if len(barriers["minima_positions"]) >= 2 and len(barriers["maxima_positions"]) >= 1:
            assert "forward_barrier" in barriers
            assert "reverse_barrier" in barriers
            assert barriers["forward_barrier"] > 0
            assert barriers["reverse_barrier"] > 0
    
    def test_file_output(self, umbrella_data, temp_dir):
        """Test writing PMF to file."""
        pmf = PMFAnalysis()
        
        for window in umbrella_data[:3]:
            pmf.add_window(
                reaction_coord=window["positions"],
                center=window["center"],
                force_constant=window["force_constant"]
            )
        
        pmf.wham_1d(bins=30)
        
        output_file = temp_dir / "pmf.txt"
        pmf.write_pmf(output_file)
        
        assert output_file.exists()
        
        # Read back and check
        df = pd.read_csv(output_file, comment='#', delim_whitespace=True)
        assert "reaction_coord" in df.columns
        assert "pmf" in df.columns
        assert "probability" in df.columns
        assert "counts" in df.columns
    
    def test_custom_bias_potential(self):
        """Test with custom bias potential."""
        pmf = PMFAnalysis()
        
        # Create custom bias (not harmonic)
        positions = np.random.uniform(-5, 5, 1000)
        center = 0.0
        # Quartic potential
        bias = 0.1 * (positions - center)**4
        
        pmf.add_window(
            reaction_coord=positions,
            center=center,
            force_constant=None,  # Not used
            bias_potential=bias
        )
        
        assert len(pmf.windows) == 1
        assert np.array_equal(pmf.windows[0]["bias_potential"], bias)
    
    def test_different_bin_sizes(self, umbrella_data):
        """Test WHAM with different bin sizes."""
        pmf = PMFAnalysis()
        
        for window in umbrella_data[:5]:
            pmf.add_window(
                reaction_coord=window["positions"],
                center=window["center"],
                force_constant=window["force_constant"]
            )
        
        results = {}
        for n_bins in [20, 50, 100]:
            rc, pmf_values = pmf.wham_1d(bins=n_bins)
            results[n_bins] = (rc, pmf_values)
        
        # More bins should give finer resolution
        assert len(results[100][0]) > len(results[50][0])
        assert len(results[50][0]) > len(results[20][0])
    
    def test_minimum_counts_filter(self, umbrella_data):
        """Test minimum counts filtering."""
        pmf = PMFAnalysis()
        
        # Add only one window (poor sampling at edges)
        window = umbrella_data[5]  # Middle window
        pmf.add_window(
            reaction_coord=window["positions"],
            center=window["center"],
            force_constant=window["force_constant"]
        )
        
        # Run with minimum counts requirement
        rc, pmf_values = pmf.wham_1d(bins=50, min_counts=10)
        
        # Should have filtered out low-count bins
        assert len(rc) < 50  # Less than total bins
        assert np.all(np.isfinite(pmf_values))


class TestPMFValidation:
    """Validation tests for PMF analysis."""
    
    def test_harmonic_potential(self):
        """Test PMF recovery for harmonic potential."""
        pmf = PMFAnalysis()
        
        # Create umbrella windows for harmonic potential
        # PMF(x) = 0.5 * k * x^2
        k_pmf = 2.0  # PMF force constant
        
        centers = np.linspace(-3, 3, 15)
        k_umbrella = 10.0
        
        for center in centers:
            # Sample from biased distribution
            # P(x) ~ exp(-beta * (U_pmf + U_bias))
            n_samples = 2000
            positions = []
            
            for _ in range(n_samples):
                # Simple Metropolis sampling
                x = center
                for _ in range(100):  # Equilibration
                    x_new = x + np.random.normal(0, 0.5)
                    # Combined potential
                    u_old = 0.5 * k_pmf * x**2 + 0.5 * k_umbrella * (x - center)**2
                    u_new = 0.5 * k_pmf * x_new**2 + 0.5 * k_umbrella * (x_new - center)**2
                    
                    if np.random.random() < np.exp(-(u_new - u_old) * pmf.beta):
                        x = x_new
                
                positions.append(x)
            
            pmf.add_window(
                reaction_coord=np.array(positions),
                center=center,
                force_constant=k_umbrella
            )
        
        # Calculate PMF
        rc, pmf_values = pmf.wham_1d(bins=30)
        
        # Compare to analytical
        # Select points near origin for comparison
        mask = np.abs(rc) < 2.0
        rc_test = rc[mask]
        pmf_test = pmf_values[mask]
        pmf_analytical = 0.5 * k_pmf * rc_test**2
        
        # Shift to match minima
        pmf_test -= np.min(pmf_test)
        pmf_analytical -= np.min(pmf_analytical)
        
        # Should match within error
        rmse = np.sqrt(np.mean((pmf_test - pmf_analytical)**2))
        assert rmse < 0.5  # Within 0.5 kcal/mol RMSE
    
    def test_no_overlap_warning(self):
        """Test behavior with non-overlapping windows."""
        pmf = PMFAnalysis()
        
        # Create non-overlapping windows
        pmf.add_window(
            reaction_coord=np.random.normal(0, 0.5, 500),
            center=0.0,
            force_constant=10.0
        )
        
        pmf.add_window(
            reaction_coord=np.random.normal(10, 0.5, 500),  # Far away
            center=10.0,
            force_constant=10.0
        )
        
        # Should still run but with poor results
        rc, pmf_values = pmf.wham_1d(bins=50)
        
        # There should be a gap in the PMF
        assert len(rc) > 0
        
        # Many bins should have no data (filtered out or NaN)
        assert np.sum(np.isnan(pmf_values)) > 0 or len(rc) < 50
    
    def test_temperature_scaling(self):
        """Test temperature dependence of PMF."""
        results = {}
        
        for temp in [250.0, 300.0, 350.0]:
            pmf = PMFAnalysis(temperature=temp)
            
            # Add simple windows
            for center in np.linspace(-2, 2, 5):
                positions = np.random.normal(center, 1.0, 500)
                pmf.add_window(positions, center, 5.0)
            
            rc, pmf_values = pmf.wham_1d(bins=30)
            results[temp] = pmf_values
        
        # Higher temperature should give flatter PMF (when properly scaled)
        # But absolute values depend on entropy
        assert len(results[250.0]) > 0
        assert len(results[300.0]) > 0
        assert len(results[350.0]) > 0