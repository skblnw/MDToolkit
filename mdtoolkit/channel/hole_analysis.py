"""HOLE program integration for channel analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import subprocess
import tempfile
import MDAnalysis as mda
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings

logger = logging.getLogger(__name__)


class HOLEAnalysis:
    """Interface to HOLE program for channel/pore analysis.
    
    HOLE analyzes channels and cavities in molecular structures,
    particularly useful for ion channels, transporters, and pores.
    """
    
    def __init__(
        self,
        hole_executable: str = "hole",
        sphpdb_executable: str = "sph_process"
    ):
        """Initialize HOLE analysis.
        
        Args:
            hole_executable: Path to HOLE executable
            sphpdb_executable: Path to sph_process executable
        """
        self.hole_exe = hole_executable
        self.sphpdb_exe = sphpdb_executable
        self.profiles = []
        self.current_profile = None
        
        # Check if HOLE is available
        self._check_hole_installation()
    
    def _check_hole_installation(self):
        """Check if HOLE is properly installed."""
        try:
            result = subprocess.run(
                [self.hole_exe],
                input=b"\n",
                capture_output=True,
                timeout=2
            )
            if result.returncode not in [0, 1]:
                warnings.warn(f"HOLE may not be properly installed: return code {result.returncode}")
        except FileNotFoundError:
            warnings.warn(
                f"HOLE executable not found at '{self.hole_exe}'. "
                "Please install HOLE and/or specify the correct path."
            )
        except subprocess.TimeoutExpired:
            pass  # HOLE is interactive, timeout is expected
        except Exception as e:
            warnings.warn(f"Error checking HOLE installation: {e}")
    
    def run_hole(
        self,
        structure: Union[str, Path, mda.Universe],
        start_point: Optional[Tuple[float, float, float]] = None,
        end_point: Optional[Tuple[float, float, float]] = None,
        cvect: Optional[Tuple[float, float, float]] = None,
        radius: float = 5.0,
        sample_distance: float = 0.2,
        output_prefix: str = "hole"
    ) -> Dict:
        """Run HOLE analysis on a structure.
        
        Args:
            structure: Path to PDB file or MDAnalysis Universe
            start_point: Starting point for channel search (x, y, z)
            end_point: End point for channel direction
            cvect: Channel vector (alternative to end_point)
            radius: Maximum radius to explore (Angstroms)
            sample_distance: Distance between sample points
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with analysis results
        """
        # Handle input structure
        if isinstance(structure, (str, Path)):
            pdb_file = Path(structure)
        else:
            # Write MDAnalysis universe to temporary PDB
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
                structure.atoms.write(tmp.name)
                pdb_file = Path(tmp.name)
        
        # Determine channel search parameters
        if start_point is None:
            # Try to guess from structure center
            if isinstance(structure, mda.Universe):
                start_point = structure.atoms.center_of_mass()
            else:
                u = mda.Universe(str(pdb_file))
                start_point = u.atoms.center_of_mass()
            logger.info(f"Using center of mass as start point: {start_point}")
        
        if cvect is None:
            if end_point is not None:
                cvect = np.array(end_point) - np.array(start_point)
                cvect = cvect / np.linalg.norm(cvect)
            else:
                cvect = (0, 0, 1)  # Default to z-axis
        
        # Create HOLE input file
        hole_input = self._create_hole_input(
            pdb_file=pdb_file,
            start_point=start_point,
            cvect=cvect,
            radius=radius,
            sample_distance=sample_distance,
            output_prefix=output_prefix
        )
        
        # Run HOLE
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as inp_file:
                inp_file.write(hole_input)
                inp_file.flush()
                
                result = subprocess.run(
                    [self.hole_exe],
                    stdin=open(inp_file.name),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    logger.error(f"HOLE failed: {result.stderr}")
                    raise RuntimeError(f"HOLE analysis failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            logger.error("HOLE analysis timed out")
            raise RuntimeError("HOLE analysis timed out after 60 seconds")
        
        # Parse results
        results = self._parse_hole_output(output_prefix)
        
        # Store profile
        self.current_profile = results
        self.profiles.append(results)
        
        # Clean up temporary files if needed
        if not isinstance(structure, (str, Path)):
            pdb_file.unlink()
        
        return results
    
    def _create_hole_input(
        self,
        pdb_file: Path,
        start_point: Tuple[float, float, float],
        cvect: Tuple[float, float, float],
        radius: float,
        sample_distance: float,
        output_prefix: str
    ) -> str:
        """Create HOLE input file content.
        
        Args:
            pdb_file: Path to PDB file
            start_point: Starting point
            cvect: Channel vector
            radius: Maximum radius
            sample_distance: Sample distance
            output_prefix: Output prefix
            
        Returns:
            HOLE input file content
        """
        input_text = f"""
! HOLE input file generated by MDToolkit
coord {pdb_file}
radius ~/hole2/rad/simple.rad
sphpdb {output_prefix}.sph
pltout {output_prefix}.plt
endrad {radius:.1f}
sample {sample_distance:.2f}
cpoint {start_point[0]:.3f} {start_point[1]:.3f} {start_point[2]:.3f}
cvect {cvect[0]:.3f} {cvect[1]:.3f} {cvect[2]:.3f}
"""
        return input_text
    
    def _parse_hole_output(self, output_prefix: str) -> Dict:
        """Parse HOLE output files.
        
        Args:
            output_prefix: Prefix used for output files
            
        Returns:
            Dictionary with parsed results
        """
        results = {
            'prefix': output_prefix,
            'profile': None,
            'min_radius': None,
            'min_radius_position': None,
            'length': None
        }
        
        # Parse the .plt file (profile data)
        plt_file = Path(f"{output_prefix}.plt")
        if plt_file.exists():
            profile_data = []
            
            with open(plt_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                distance = float(parts[0])
                                radius = float(parts[1])
                                profile_data.append((distance, radius))
                            except ValueError:
                                continue
            
            if profile_data:
                profile_df = pd.DataFrame(profile_data, columns=['distance', 'radius'])
                results['profile'] = profile_df
                results['min_radius'] = profile_df['radius'].min()
                min_idx = profile_df['radius'].idxmin()
                results['min_radius_position'] = profile_df.loc[min_idx, 'distance']
                results['length'] = profile_df['distance'].max() - profile_df['distance'].min()
        
        return results
    
    def analyze_trajectory(
        self,
        universe: mda.Universe,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        **hole_kwargs
    ) -> pd.DataFrame:
        """Analyze channel profile over trajectory.
        
        Args:
            universe: MDAnalysis Universe
            start: Starting frame
            stop: Stopping frame
            step: Frame step
            **hole_kwargs: Additional arguments for run_hole
            
        Returns:
            DataFrame with time series of channel properties
        """
        results = []
        
        for ts in universe.trajectory[start:stop:step]:
            frame_results = self.run_hole(universe, **hole_kwargs)
            
            if frame_results['profile'] is not None:
                results.append({
                    'frame': ts.frame,
                    'time': ts.time,
                    'min_radius': frame_results['min_radius'],
                    'min_radius_position': frame_results['min_radius_position'],
                    'length': frame_results['length']
                })
            
            logger.info(f"Analyzed frame {ts.frame}/{universe.trajectory.n_frames}")
        
        return pd.DataFrame(results)
    
    def plot_profile(
        self,
        profile: Optional[pd.DataFrame] = None,
        output_file: Optional[Union[str, Path]] = None,
        show_classification: bool = True
    ):
        """Plot pore radius profile.
        
        Args:
            profile: Profile DataFrame (uses current if None)
            output_file: Optional output file
            show_classification: Show radius classification zones
        """
        if profile is None:
            if self.current_profile and self.current_profile['profile'] is not None:
                profile = self.current_profile['profile']
            else:
                raise ValueError("No profile data available")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot profile
        ax.plot(profile['distance'], profile['radius'], 'b-', linewidth=2)
        
        # Add classification zones if requested
        if show_classification:
            ax.axhspan(0, 1.15, alpha=0.3, color='red', label='Too narrow (<1.15Å)')
            ax.axhspan(1.15, 2.3, alpha=0.3, color='yellow', label='Single water (1.15-2.3Å)')
            ax.axhspan(2.3, 100, alpha=0.3, color='green', label='Bulk water (>2.3Å)')
        
        # Mark minimum radius
        min_radius = profile['radius'].min()
        min_idx = profile['radius'].idxmin()
        min_pos = profile.loc[min_idx, 'distance']
        
        ax.plot(min_pos, min_radius, 'ro', markersize=10,
               label=f'Min radius: {min_radius:.2f}Å at {min_pos:.2f}Å')
        
        ax.set_xlabel('Distance along channel (Å)')
        ax.set_ylabel('Pore radius (Å)')
        ax.set_title('Channel/Pore Radius Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        ax.set_ylim(0, min(profile['radius'].max() * 1.1, 20))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved profile plot to {output_file}")
        else:
            plt.show()
    
    def plot_channel_3d(
        self,
        sph_file: Optional[Union[str, Path]] = None,
        structure: Optional[mda.Universe] = None,
        output_file: Optional[Union[str, Path]] = None
    ):
        """Create 3D visualization of channel.
        
        Args:
            sph_file: Path to .sph file from HOLE
            structure: Optional structure for overlay
            output_file: Optional output file
        """
        if sph_file is None:
            sph_file = Path(f"{self.current_profile['prefix']}.sph")
        
        if not Path(sph_file).exists():
            logger.warning(f"SPH file not found: {sph_file}")
            return
        
        # Parse sphere file
        spheres = []
        with open(sph_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    radius = float(line[60:66])
                    spheres.append((x, y, z, radius))
        
        if not spheres:
            logger.warning("No spheres found in SPH file")
            return
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot spheres
        for x, y, z, r in spheres:
            # Color by radius
            if r < 1.15:
                color = 'red'
            elif r < 2.3:
                color = 'yellow'
            else:
                color = 'green'
            
            ax.scatter(x, y, z, s=r*50, c=color, alpha=0.6)
        
        # Add structure if provided
        if structure is not None:
            # Plot CA atoms
            ca_atoms = structure.select_atoms("protein and name CA")
            if len(ca_atoms) > 0:
                coords = ca_atoms.positions
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c='gray', s=10, alpha=0.3)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Channel/Pore 3D Visualization')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D plot to {output_file}")
        else:
            plt.show()
    
    def get_bottleneck_residues(
        self,
        structure: mda.Universe,
        profile: Optional[pd.DataFrame] = None,
        cutoff: float = 5.0
    ) -> List[Tuple[int, str, float]]:
        """Identify residues at channel bottleneck.
        
        Args:
            structure: MDAnalysis Universe
            profile: Profile DataFrame
            cutoff: Distance cutoff for residue identification
            
        Returns:
            List of (resid, resname, distance) tuples
        """
        if profile is None:
            if self.current_profile and self.current_profile['profile'] is not None:
                profile = self.current_profile['profile']
            else:
                raise ValueError("No profile data available")
        
        # Find bottleneck position
        min_idx = profile['radius'].idxmin()
        min_pos = profile.loc[min_idx, 'distance']
        
        # This would require the actual 3D coordinates from HOLE
        # For now, return placeholder
        logger.warning("Bottleneck residue identification requires SPH file parsing")
        
        return []
    
    def write_profile(self, output_file: Union[str, Path], profile: Optional[pd.DataFrame] = None):
        """Write profile data to file.
        
        Args:
            output_file: Output file path
            profile: Profile DataFrame (uses current if None)
        """
        if profile is None:
            if self.current_profile and self.current_profile['profile'] is not None:
                profile = self.current_profile['profile']
            else:
                raise ValueError("No profile data available")
        
        output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("# HOLE Channel/Pore Profile\n")
            f.write(f"# Minimum radius: {profile['radius'].min():.3f} Å\n")
            f.write(f"# Channel length: {profile['distance'].max() - profile['distance'].min():.3f} Å\n")
            f.write("#\n")
            f.write("# Distance(Å) Radius(Å)\n")
        
        profile.to_csv(output_file, mode='a', index=False, sep=' ', header=False)
        
        logger.info(f"Wrote profile to {output_file}")
    
    def __repr__(self) -> str:
        """String representation."""
        n_profiles = len(self.profiles)
        return f"HOLEAnalysis(profiles={n_profiles})"