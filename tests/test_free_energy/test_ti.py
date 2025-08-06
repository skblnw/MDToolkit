"""Tests for Thermodynamic Integration analysis module."""

import pytest
import numpy as np
from scipy import integrate
from pathlib import Path

from mdtoolkit.free_energy import TIAnalysis


class TestTIAnalysis:
    """Test TI analysis functionality."""
    
    def test_initialization(self):
        """Test TIAnalysis initialization."""
        ti = TIAnalysis(temperature=300.0)
        
        assert ti.temperature == 300.0
        assert ti.beta == pytest.approx(1.0 / (0.0019872041 * 300.0))
        assert len(ti.lambda_values) == 0
        assert ti.free_energy is None
        assert ti.error is None
    
    def test_add_lambda_window(self, ti_data):
        """Test adding lambda windows."""
        ti = TIAnalysis()
        
        for lam, data in ti_data.items():
            ti.add_lambda_window(lam, data["dudl"])
        
        assert len(ti.lambda_values) == len(ti_data)
        assert all(lam in ti.dudl_values for lam in ti_data.keys())
        
        # Check that means are calculated correctly
        for lam, data in ti_data.items():
            stored = ti.dudl_values[lam]
            assert "mean" in stored
            assert "error" in stored
            assert stored["mean"] == pytest.approx(np.mean(data["dudl"]), abs=0.5)
    
    def test_trapezoid_integration(self, ti_data):
        """Test trapezoidal integration."""
        ti = TIAnalysis()
        
        # Add windows
        for lam, data in ti_data.items():
            ti.add_lambda_window(lam, data["dudl"])
        
        # Integrate
        free_energy, error = ti.integrate_ti(method="trapezoid")
        
        assert isinstance(free_energy, float)
        assert isinstance(error, float)
        assert np.isfinite(free_energy)
        assert error > 0
        
        # Check against scipy
        lambdas = sorted(ti.lambda_values)
        means = [ti.dudl_values[lam]["mean"] for lam in lambdas]
        scipy_result = integrate.trapezoid(means, lambdas)
        
        assert free_energy == pytest.approx(scipy_result, rel=1e-5)
    
    def test_simpson_integration(self):
        """Test Simpson's rule integration."""
        ti = TIAnalysis()
        
        # Create data with odd number of points
        lambdas = np.linspace(0, 1, 11)  # 11 points for Simpson's rule
        
        for lam in lambdas:
            dudl = np.random.normal(10 * (1 - 2*lam), 1.0, 500)
            ti.add_lambda_window(lam, dudl)
        
        free_energy, error = ti.integrate_ti(method="simpson")
        
        assert isinstance(free_energy, float)
        assert isinstance(error, float)
        assert np.isfinite(free_energy)
        assert error > 0
    
    def test_spline_integration(self, ti_data):
        """Test spline integration."""
        ti = TIAnalysis()
        
        for lam, data in ti_data.items():
            ti.add_lambda_window(lam, data["dudl"])
        
        free_energy, error = ti.integrate_ti(method="spline", spline_order=3)
        
        assert isinstance(free_energy, float)
        assert isinstance(error, float)
        assert np.isfinite(free_energy)
        assert error > 0
    
    def test_overlap_analysis(self, ti_data):
        """Test phase space overlap analysis."""
        ti = TIAnalysis()
        
        for lam, data in ti_data.items():
            ti.add_lambda_window(lam, data["dudl"])
        
        overlap_results = ti.analyze_overlap()
        
        assert "overlaps" in overlap_results
        assert "mean_overlap" in overlap_results
        assert "min_overlap" in overlap_results
        assert "lambda_pairs" in overlap_results
        
        # Overlaps should be between 0 and 1
        overlaps = overlap_results["overlaps"]
        assert all(0 <= o <= 1 for o in overlaps)
        assert 0 <= overlap_results["mean_overlap"] <= 1
    
    def test_weighted_averaging(self):
        """Test weighted averaging of dU/dλ."""
        ti = TIAnalysis()
        
        # Create data with weights
        dudl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        
        ti.add_lambda_window(0.5, dudl, weights=weights)
        
        # Check weighted mean
        expected_mean = np.average(dudl, weights=weights)
        assert ti.dudl_values[0.5]["mean"] == pytest.approx(expected_mean)
    
    def test_file_output(self, ti_data, temp_dir):
        """Test writing results to file."""
        ti = TIAnalysis()
        
        for lam, data in ti_data.items():
            ti.add_lambda_window(lam, data["dudl"])
        
        ti.integrate_ti(method="trapezoid")
        
        output_file = temp_dir / "ti_results.txt"
        ti.write_results(output_file)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            content = f.read()
            assert "Thermodynamic Integration" in content
            assert "Temperature: 300.0 K" in content
            assert "Free Energy:" in content
    
    def test_dataframe_output(self, ti_data):
        """Test pandas DataFrame output."""
        ti = TIAnalysis()
        
        for lam, data in ti_data.items():
            ti.add_lambda_window(lam, data["dudl"])
        
        ti.integrate_ti()
        df = ti.get_results_dataframe()
        
        assert "lambda" in df.columns
        assert "dudl_mean" in df.columns
        assert "dudl_error" in df.columns
        assert "n_samples" in df.columns
        
        assert len(df) == len(ti_data)
        assert df["lambda"].is_monotonic_increasing
        
        # Check attributes
        assert "free_energy" in df.attrs
        assert "error" in df.attrs
    
    @pytest.mark.parametrize("n_windows", [3, 5, 11, 21])
    def test_different_window_numbers(self, n_windows):
        """Test with different numbers of lambda windows."""
        ti = TIAnalysis()
        
        lambdas = np.linspace(0, 1, n_windows)
        for lam in lambdas:
            dudl = np.random.normal(5 * (1 - lam), 0.5, 100)
            ti.add_lambda_window(lam, dudl)
        
        free_energy, error = ti.integrate_ti()
        
        assert np.isfinite(free_energy)
        assert error > 0
    
    def test_integration_methods_consistency(self, ti_data):
        """Test that different integration methods give similar results."""
        results = {}
        
        for method in ["trapezoid", "spline"]:
            ti = TIAnalysis()
            for lam, data in ti_data.items():
                ti.add_lambda_window(lam, data["dudl"])
            
            free_energy, error = ti.integrate_ti(method=method)
            results[method] = free_energy
        
        # Methods should agree within reasonable tolerance
        values = list(results.values())
        assert np.std(values) < 1.0  # Within 1 kcal/mol


class TestTIValidation:
    """Validation tests for TI analysis."""
    
    def test_linear_perturbation(self):
        """Test with linear perturbation (analytical result)."""
        ti = TIAnalysis()
        
        # For linear dU/dλ = a + b*λ, integral is a + b/2
        a, b = 5.0, -10.0
        expected = a + b/2  # Analytical result
        
        lambdas = np.linspace(0, 1, 21)
        for lam in lambdas:
            dudl_mean = a + b * lam
            # Add small noise
            dudl = np.ones(100) * dudl_mean + np.random.normal(0, 0.01, 100)
            ti.add_lambda_window(lam, dudl)
        
        free_energy, _ = ti.integrate_ti(method="trapezoid")
        
        assert free_energy == pytest.approx(expected, abs=0.1)
    
    def test_quadratic_perturbation(self):
        """Test with quadratic perturbation."""
        ti = TIAnalysis()
        
        # For dU/dλ = a*λ^2, integral is a/3
        a = 12.0
        expected = a / 3
        
        lambdas = np.linspace(0, 1, 21)
        for lam in lambdas:
            dudl_mean = a * lam**2
            dudl = np.ones(100) * dudl_mean + np.random.normal(0, 0.01, 100)
            ti.add_lambda_window(lam, dudl)
        
        # Spline should handle quadratic well
        free_energy, _ = ti.integrate_ti(method="spline", spline_order=3)
        
        assert free_energy == pytest.approx(expected, abs=0.2)
    
    def test_insufficient_windows(self):
        """Test error handling with insufficient windows."""
        ti = TIAnalysis()
        
        # Add only one window
        ti.add_lambda_window(0.5, np.random.normal(0, 1, 100))
        
        with pytest.raises(Exception):
            ti.integrate_ti()
    
    def test_bootstrap_error_estimation(self):
        """Test bootstrap error estimation consistency."""
        ti = TIAnalysis()
        
        # Create reproducible data
        np.random.seed(42)
        lambdas = np.linspace(0, 1, 11)
        
        for lam in lambdas:
            dudl = np.random.normal(5 * (1 - lam), 1.0, 500)
            ti.add_lambda_window(lam, dudl)
        
        free_energy1, error1 = ti.integrate_ti()
        
        # Reset and repeat with same seed
        np.random.seed(42)
        ti2 = TIAnalysis()
        
        for lam in lambdas:
            dudl = np.random.normal(5 * (1 - lam), 1.0, 500)
            ti2.add_lambda_window(lam, dudl)
        
        free_energy2, error2 = ti2.integrate_ti()
        
        # Results should be identical with same seed
        assert free_energy1 == pytest.approx(free_energy2)
        assert error1 == pytest.approx(error2, rel=0.1)