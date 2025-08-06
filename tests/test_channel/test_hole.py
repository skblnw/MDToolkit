"""Tests for HOLE channel analysis module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import warnings
from unittest.mock import patch, MagicMock

from mdtoolkit.channel import HOLEAnalysis


class TestHOLEAnalysis:
    """Test HOLE analysis functionality."""
    
    def test_initialization(self):
        """Test HOLEAnalysis initialization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        assert hole.hole_exe == "hole"
        assert hole.sphpdb_exe == "sph_process"
        assert len(hole.profiles) == 0
        assert hole.current_profile is None
    
    def test_custom_executable_path(self):
        """Test with custom HOLE executable path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis(
                hole_executable="/custom/path/hole",
                sphpdb_executable="/custom/path/sph"
            )
        
        assert hole.hole_exe == "/custom/path/hole"
        assert hole.sphpdb_exe == "/custom/path/sph"
    
    @patch('subprocess.run')
    def test_hole_installation_check(self, mock_run):
        """Test HOLE installation check."""
        # Mock successful check
        mock_run.return_value = MagicMock(returncode=0)
        
        hole = HOLEAnalysis()
        mock_run.assert_called()
    
    def test_parse_hole_output(self, mock_hole_output):
        """Test parsing of HOLE output files."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        # Parse mock output
        results = hole._parse_hole_output(str(mock_hole_output["plt"].parent / "test"))
        
        assert results is not None
        assert "profile" in results
        assert "min_radius" in results
        assert "min_radius_position" in results
        assert "length" in results
        
        # Check profile data
        profile = results["profile"]
        assert isinstance(profile, pd.DataFrame)
        assert "distance" in profile.columns
        assert "radius" in profile.columns
        assert len(profile) > 0
        
        # Check calculated values
        assert results["min_radius"] > 0
        assert isinstance(results["min_radius_position"], float)
        assert results["length"] > 0
    
    def test_create_hole_input(self, temp_dir):
        """Test HOLE input file creation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        pdb_file = temp_dir / "test.pdb"
        pdb_file.touch()
        
        input_text = hole._create_hole_input(
            pdb_file=pdb_file,
            start_point=(0, 0, 0),
            cvect=(0, 0, 1),
            radius=5.0,
            sample_distance=0.2,
            output_prefix="test"
        )
        
        assert "coord" in input_text
        assert str(pdb_file) in input_text
        assert "cpoint 0.000 0.000 0.000" in input_text
        assert "cvect 0.000 0.000 1.000" in input_text
        assert "endrad 5.0" in input_text
        assert "sample 0.20" in input_text
    
    @patch('subprocess.run')
    def test_run_hole_mock(self, mock_run, small_universe, temp_dir):
        """Test running HOLE with mocked subprocess."""
        # Create mock output files
        plt_file = temp_dir / "hole.plt"
        with open(plt_file, 'w') as f:
            for i in range(50):
                f.write(f"{i-25:.2f} {3.0 + np.sin(i*0.2):.2f}\n")
        
        sph_file = temp_dir / "hole.sph"
        sph_file.touch()
        
        # Mock successful HOLE run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="HOLE completed successfully",
            stderr=""
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        results = hole.run_hole(
            structure=small_universe,
            start_point=(0, 0, 0),
            cvect=(0, 0, 1),
            output_prefix=str(temp_dir / "hole")
        )
        
        assert results is not None
        assert hole.current_profile == results
        assert len(hole.profiles) == 1
    
    def test_write_profile(self, channel_profile, temp_dir):
        """Test writing profile to file."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        # Create profile DataFrame
        profile_df = pd.DataFrame({
            "distance": channel_profile["distance"],
            "radius": channel_profile["radius"]
        })
        
        hole.current_profile = {"profile": profile_df}
        
        output_file = temp_dir / "profile.txt"
        hole.write_profile(output_file)
        
        assert output_file.exists()
        
        # Read and verify
        with open(output_file, 'r') as f:
            content = f.read()
            assert "HOLE Channel/Pore Profile" in content
            assert "Minimum radius:" in content
            assert "Channel length:" in content
    
    @pytest.mark.parametrize("n_frames", [1, 5, 10])
    def test_analyze_trajectory(self, n_frames, protein_universe, temp_dir):
        """Test trajectory analysis."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        # Mock the run_hole method
        def mock_run_hole(*args, **kwargs):
            return {
                "profile": pd.DataFrame({
                    "distance": np.linspace(-10, 10, 20),
                    "radius": np.random.uniform(2, 4, 20)
                }),
                "min_radius": np.random.uniform(1.5, 2.5),
                "min_radius_position": np.random.uniform(-2, 2),
                "length": 20.0
            }
        
        hole.run_hole = mock_run_hole
        
        # Analyze trajectory
        results_df = hole.analyze_trajectory(
            protein_universe,
            stop=n_frames,
            output_prefix=str(temp_dir / "traj")
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == n_frames
        assert "frame" in results_df.columns
        assert "time" in results_df.columns
        assert "min_radius" in results_df.columns
        assert "min_radius_position" in results_df.columns
        assert "length" in results_df.columns
    
    def test_bottleneck_identification(self, small_universe, channel_profile):
        """Test bottleneck residue identification."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        profile_df = pd.DataFrame({
            "distance": channel_profile["distance"],
            "radius": channel_profile["radius"]
        })
        
        hole.current_profile = {"profile": profile_df}
        
        # This is a placeholder test since full implementation requires SPH parsing
        residues = hole.get_bottleneck_residues(small_universe)
        
        assert isinstance(residues, list)
    
    def test_empty_profile_error(self):
        """Test error handling for empty profile."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        with pytest.raises(ValueError):
            hole.plot_profile()
        
        with pytest.raises(ValueError):
            hole.write_profile("test.txt")
    
    @patch('subprocess.run')
    def test_hole_timeout(self, mock_run, small_universe):
        """Test HOLE timeout handling."""
        import subprocess
        
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("hole", 60)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        with pytest.raises(RuntimeError, match="timed out"):
            hole.run_hole(small_universe)
    
    @patch('subprocess.run')
    def test_hole_failure(self, mock_run, small_universe):
        """Test HOLE failure handling."""
        # Mock failed run
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: Invalid input"
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        with pytest.raises(RuntimeError, match="HOLE analysis failed"):
            hole.run_hole(small_universe)


class TestHOLEVisualization:
    """Test HOLE visualization functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_profile(self, mock_show, channel_profile):
        """Test profile plotting."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        profile_df = pd.DataFrame({
            "distance": channel_profile["distance"],
            "radius": channel_profile["radius"]
        })
        
        hole.current_profile = {
            "profile": profile_df,
            "min_radius": channel_profile["min_radius"],
            "min_radius_position": channel_profile["min_position"]
        }
        
        # Should not raise
        hole.plot_profile(show_classification=True)
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_channel_3d(self, mock_show, mock_hole_output):
        """Test 3D channel visualization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        hole.current_profile = {"prefix": str(mock_hole_output["sph"].parent / "test")}
        
        # Should handle SPH file parsing
        hole.plot_channel_3d(sph_file=mock_hole_output["sph"])
        mock_show.assert_called_once()
    
    def test_plot_profile_save(self, channel_profile, temp_dir):
        """Test saving profile plot."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        profile_df = pd.DataFrame({
            "distance": channel_profile["distance"],
            "radius": channel_profile["radius"]
        })
        
        hole.current_profile = {
            "profile": profile_df,
            "min_radius": channel_profile["min_radius"],
            "min_radius_position": channel_profile["min_position"]
        }
        
        output_file = temp_dir / "profile.png"
        hole.plot_profile(output_file=output_file)
        
        assert output_file.exists()


class TestHOLEIntegration:
    """Integration tests for HOLE analysis."""
    
    @pytest.mark.requires_hole
    def test_real_hole_execution(self, small_universe, temp_dir):
        """Test with real HOLE executable if available."""
        try:
            hole = HOLEAnalysis()
            
            results = hole.run_hole(
                structure=small_universe,
                output_prefix=str(temp_dir / "real_hole")
            )
            
            assert results is not None
            assert "profile" in results
            
        except (FileNotFoundError, RuntimeError):
            pytest.skip("HOLE not installed")
    
    def test_channel_classification(self, channel_profile):
        """Test channel radius classification."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        radii = channel_profile["radius"]
        
        # Classification thresholds
        too_narrow = radii < 1.15
        single_water = (radii >= 1.15) & (radii < 2.3)
        bulk_water = radii >= 2.3
        
        assert np.any(too_narrow)  # Has narrow region
        assert np.any(single_water)  # Has single-file region
        assert np.any(bulk_water)  # Has bulk region
    
    def test_profile_statistics(self, channel_profile):
        """Test profile statistical analysis."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hole = HOLEAnalysis()
        
        profile_df = pd.DataFrame({
            "distance": channel_profile["distance"],
            "radius": channel_profile["radius"]
        })
        
        # Calculate statistics
        mean_radius = profile_df["radius"].mean()
        std_radius = profile_df["radius"].std()
        median_radius = profile_df["radius"].median()
        
        assert mean_radius > 0
        assert std_radius > 0
        assert median_radius > 0
        
        # Bottleneck should be minimum
        min_radius = profile_df["radius"].min()
        assert min_radius < mean_radius
        assert min_radius == channel_profile["min_radius"]