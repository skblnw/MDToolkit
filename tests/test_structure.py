"""Tests for structure analysis module."""

import pytest
import numpy as np
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD

from mdtoolkit.structure import (
    RMSDAnalysis,
    calculate_rmsd,
    calculate_rmsf,
    ContactAnalysis,
    NativeContacts,
    HydrogenBonds,
)


class TestRMSDAnalysis:
    """Test RMSD analysis functionality."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_rmsd_calculation(self, universe):
        """Test basic RMSD calculation."""
        rmsd = calculate_rmsd(universe, selection="name CA")
        
        assert isinstance(rmsd, np.ndarray)
        assert rmsd.shape[1] == 3  # frame, time, rmsd
        assert np.all(rmsd[:, 2] >= 0)  # RMSD should be non-negative
    
    def test_rmsd_analysis_class(self, universe):
        """Test RMSDAnalysis class."""
        analysis = RMSDAnalysis(
            universe,
            align_selection="name CA",
            analysis_selections={"ca": "name CA", "backbone": "backbone"}
        )
        
        results = analysis.run()
        
        assert "ca" in results
        assert "backbone" in results
        assert "time" in results["ca"]
        assert "rmsd" in results["ca"]
    
    def test_rmsf_calculation(self, universe):
        """Test RMSF calculation."""
        rmsf = calculate_rmsf(universe, selection="name CA")
        
        assert isinstance(rmsf, np.ndarray)
        assert len(rmsf) == len(universe.select_atoms("name CA"))
        assert np.all(rmsf >= 0)


class TestContactAnalysis:
    """Test contact analysis functionality."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_contact_analysis(self, universe):
        """Test basic contact analysis."""
        analysis = ContactAnalysis(
            universe,
            selection1="resid 1-10 and name CA",
            selection2="resid 50-60 and name CA",
            cutoff=8.0
        )
        
        results = analysis.run(stop=5)  # Run on first 5 frames
        
        assert "frames" in results
        assert "n_contacts" in results
        assert len(results["frames"]) == 5
        assert np.all(results["n_contacts"] >= 0)
    
    def test_native_contacts(self, universe):
        """Test native contact analysis."""
        analysis = NativeContacts(
            universe,
            selection="name CA",
            radius=8.0
        )
        
        results = analysis.run(stop=5)
        
        assert "time" in results
        assert "q" in results  # Fraction of native contacts
        assert np.all(results["q"] >= 0)
        assert np.all(results["q"] <= 1)
    
    @pytest.mark.skipif(not hasattr(mda.Universe, "select_atoms"), 
                        reason="Requires MDAnalysis with H-bond analysis")
    def test_hydrogen_bonds(self, universe):
        """Test hydrogen bond analysis."""
        analysis = HydrogenBonds(
            universe,
            distance_cutoff=3.5,
            angle_cutoff=150.0
        )
        
        results = analysis.run(stop=5)
        
        assert "time" in results
        assert "n_hbonds" in results
        assert np.all(results["n_hbonds"] >= 0)


class TestDistanceAnalysis:
    """Test distance calculations."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_distance_calculation(self, universe):
        """Test distance analysis."""
        from mdtoolkit.structure import DistanceAnalysis
        
        analysis = DistanceAnalysis(
            universe,
            selection1="resid 1 and name CA",
            selection2="resid 10 and name CA"
        )
        
        results = analysis.run(stop=5)
        
        assert "mean" in results
        assert "min" in results
        assert "max" in results
        assert len(results["mean"]) == 5


class TestGeometry:
    """Test geometric calculations."""
    
    @pytest.fixture
    def universe(self):
        """Create test universe."""
        return mda.Universe(PSF, DCD)
    
    def test_radius_of_gyration(self, universe):
        """Test radius of gyration calculation."""
        from mdtoolkit.structure import RadiusOfGyration
        
        analysis = RadiusOfGyration(universe, selection="name CA")
        rgyr = analysis.run()
        
        assert isinstance(rgyr, np.ndarray)
        assert len(rgyr) == len(universe.trajectory)
        assert np.all(rgyr > 0)
    
    def test_angle_calculation(self, universe):
        """Test angle calculation."""
        from mdtoolkit.structure import calculate_angle
        
        angles = calculate_angle(universe, 0, 1, 2)  # First three atoms
        
        assert isinstance(angles, np.ndarray)
        assert len(angles) == len(universe.trajectory)
        assert np.all(angles >= 0)
        assert np.all(angles <= 180)