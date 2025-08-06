"""Tests for dynamics analysis module."""

import pytest
import numpy as np
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD

from mdtoolkit.dynamics import (
    CorrelationAnalysis,
    PCAAnalysis,
    calculate_correlation_matrix,
    perform_pca,
    validate_pca_with_sklearn,
)


class TestCorrelationAnalysis:
    """Test correlation analysis functionality."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_correlation_matrix(self, universe):
        """Test correlation matrix calculation."""
        corr_matrix = calculate_correlation_matrix(
            universe,
            selection="name CA"
        )
        
        assert isinstance(corr_matrix, np.ndarray)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert np.all(corr_matrix >= -1)
        assert np.all(corr_matrix <= 1)
        assert np.allclose(np.diag(corr_matrix), 1.0, atol=1e-5)
    
    def test_correlation_analysis_class(self, universe):
        """Test CorrelationAnalysis class."""
        analysis = CorrelationAnalysis(
            universe,
            selection="name CA",
            align=True
        )
        
        analysis.extract_positions()
        corr_matrix = analysis.calculate_correlation_matrix(method="pearson")
        
        assert corr_matrix is not None
        assert corr_matrix.shape[0] == len(universe.select_atoms("name CA"))
    
    def test_residue_correlation(self, universe):
        """Test residue-level correlation."""
        analysis = CorrelationAnalysis(universe, selection="name CA")
        analysis.extract_positions()
        analysis.calculate_correlation_matrix()
        
        res_corr = analysis.calculate_residue_correlation()
        
        n_residues = len(universe.select_atoms("name CA").residues)
        assert res_corr.shape == (n_residues, n_residues)


class TestPCAAnalysis:
    """Test PCA analysis functionality."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_quick_pca(self, universe):
        """Test quick PCA function."""
        projections, eigenvalues, eigenvectors = perform_pca(
            universe,
            selection="name CA",
            n_components=3
        )
        
        assert projections.shape[1] == 3
        assert len(eigenvalues) == 3
        assert eigenvectors.shape[0] == 3
    
    def test_pca_analysis_class(self, universe):
        """Test PCAAnalysis class."""
        analysis = PCAAnalysis(
            universe,
            selection="name CA",
            align=True
        )
        
        mda_results = analysis.run_mda_pca()
        
        assert "variance" in mda_results
        assert "cumulated_variance" in mda_results
        assert "transformed" in mda_results
        assert mda_results["transformed"] is not None
    
    def test_pca_validation(self, universe):
        """Test PCA validation with sklearn."""
        analysis = PCAAnalysis(universe, selection="name CA")
        
        # Run both PCA implementations
        analysis.run_mda_pca()
        analysis.run_sklearn_pca(n_components=5)
        
        # Validate
        validation = analysis.validate_pca()
        
        assert "variance_match" in validation
        assert "projection_correlations" in validation
        assert len(validation["projection_correlations"]) >= 1
    
    def test_cosine_content(self, universe):
        """Test cosine content calculation."""
        analysis = PCAAnalysis(universe, selection="name CA")
        analysis.run_mda_pca()
        
        cosine_content = analysis.calculate_cosine_content(n_components=3)
        
        assert len(cosine_content) == 3
        assert np.all(cosine_content >= 0)
        assert np.all(cosine_content <= 1)


class TestCovarianceAnalysis:
    """Test covariance analysis."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_covariance_matrix(self, universe):
        """Test covariance matrix calculation."""
        from mdtoolkit.dynamics import CovarianceAnalysis
        
        analysis = CovarianceAnalysis(
            universe,
            selection="name CA",
            align=True
        )
        
        cov_matrix = analysis.calculate_covariance()
        
        assert isinstance(cov_matrix, np.ndarray)
        # Covariance matrix for 3D coordinates
        n_atoms = len(universe.select_atoms("name CA"))
        assert cov_matrix.shape == (n_atoms * 3, n_atoms * 3)
        
        # Should be symmetric
        assert np.allclose(cov_matrix, cov_matrix.T)
    
    def test_eigenvalue_decomposition(self, universe):
        """Test eigenvalue decomposition of covariance matrix."""
        from mdtoolkit.dynamics import CovarianceAnalysis
        
        analysis = CovarianceAnalysis(universe, selection="name CA")
        analysis.calculate_covariance()
        
        eigenvalues, eigenvectors = analysis.get_eigenvalues()
        
        assert len(eigenvalues) > 0
        # Eigenvalues should be sorted in descending order
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


class TestIntegration:
    """Test integration between correlation and PCA."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_correlation_pca_integration(self, universe):
        """Test that correlation and PCA work on same data."""
        # Run correlation
        corr = CorrelationAnalysis(universe, selection="name CA")
        corr.extract_positions()
        corr_matrix = corr.calculate_correlation_matrix()
        
        # Run PCA on same selection
        pca = PCAAnalysis(universe, selection="name CA")
        pca_results = pca.run_mda_pca()
        
        # Both should work with same selection
        n_atoms = len(universe.select_atoms("name CA"))
        assert corr_matrix.shape == (n_atoms, n_atoms)
        assert pca_results["transformed"] is not None