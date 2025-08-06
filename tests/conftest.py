"""Shared test fixtures and configuration for MDToolkit tests."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD, PDB_small, GRO, XTC
import warnings

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================================
# Trajectory and Universe Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_universe():
    """Small test universe for quick tests."""
    return mda.Universe(PDB_small)


@pytest.fixture
def protein_universe():
    """Protein universe with trajectory."""
    return mda.Universe(PSF, DCD)


@pytest.fixture
def trajectory_universe():
    """Universe with multiple frames for trajectory analysis."""
    u = mda.Universe(GRO, XTC)
    return u


@pytest.fixture
def membrane_universe():
    """Create a mock membrane system."""
    # Create a simple membrane-like system
    n_lipids = 64
    n_atoms_per_lipid = 50
    n_atoms = n_lipids * n_atoms_per_lipid
    
    # Create universe with fake membrane
    u = mda.Universe.empty(n_atoms, n_residues=n_lipids, 
                           atom_resindex=np.repeat(range(n_lipids), n_atoms_per_lipid),
                           trajectory=True)
    
    # Add some coordinates
    coords = np.random.randn(n_atoms, 3) * 10
    u.atoms.positions = coords
    
    # Set residue names
    for i, res in enumerate(u.residues):
        res.resname = "POPC" if i % 2 == 0 else "POPE"
    
    return u


# ============================================================================
# Free Energy Data Fixtures
# ============================================================================

@pytest.fixture
def fep_data():
    """Generate synthetic FEP data."""
    n_samples = 1000
    lambda_windows = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), 
                     (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    
    data = {}
    for i, (lambda1, lambda2) in enumerate(lambda_windows):
        # Generate forward and backward work values
        mean_work = 2.0 * (lambda2 - lambda1)  # Linear response
        forward_work = np.random.normal(mean_work, 0.5, n_samples)
        backward_work = np.random.normal(-mean_work, 0.5, n_samples)
        
        data[f"window_{i}"] = {
            "lambda_start": lambda1,
            "lambda_end": lambda2,
            "forward": forward_work,
            "backward": backward_work
        }
    
    return data


@pytest.fixture
def ti_data():
    """Generate synthetic TI data."""
    lambda_values = np.linspace(0, 1, 11)
    n_samples = 500
    
    data = {}
    for lam in lambda_values:
        # dU/dλ follows a parabolic profile
        mean_dudl = 10.0 * (1 - 2*lam)  # Parabola
        dudl_samples = np.random.normal(mean_dudl, 2.0, n_samples)
        
        data[lam] = {
            "lambda": lam,
            "dudl": dudl_samples,
            "mean": mean_dudl,
            "std": 2.0
        }
    
    return data


@pytest.fixture
def umbrella_data():
    """Generate synthetic umbrella sampling data."""
    n_windows = 10
    n_samples = 2000
    centers = np.linspace(0, 10, n_windows)
    force_constant = 10.0  # kcal/mol/Å²
    
    windows = []
    for center in centers:
        # Sample from biased distribution
        positions = np.random.normal(center, 1.0/np.sqrt(force_constant), n_samples)
        
        windows.append({
            "center": center,
            "force_constant": force_constant,
            "positions": positions
        })
    
    return windows


@pytest.fixture
def bar_data():
    """Generate synthetic data for BAR analysis."""
    n_states = 5
    n_samples = 1000
    
    states = []
    for i in range(n_states):
        # Generate reduced potentials
        potentials = {}
        for j in range(n_states):
            # Energy difference between states
            delta = abs(i - j) * 2.0
            potentials[j] = np.random.normal(delta, 1.0, n_samples)
        
        states.append({
            "id": i,
            "potentials": potentials,
            "n_samples": n_samples
        })
    
    return states


# ============================================================================
# Channel/HOLE Data Fixtures
# ============================================================================

@pytest.fixture
def channel_profile():
    """Generate synthetic channel profile data."""
    distances = np.linspace(-20, 20, 100)
    
    # Create a channel profile with a bottleneck
    radii = 5.0 + 2.0 * np.cos(distances * np.pi / 20)
    radii[40:60] = 1.5  # Bottleneck region
    
    return {
        "distance": distances,
        "radius": radii,
        "min_radius": 1.5,
        "min_position": 0.0
    }


@pytest.fixture
def mock_hole_output(temp_dir):
    """Create mock HOLE output files."""
    plt_file = temp_dir / "test.plt"
    sph_file = temp_dir / "test.sph"
    
    # Write mock PLT file
    with open(plt_file, 'w') as f:
        f.write("# HOLE profile\n")
        for i in range(100):
            dist = i * 0.5 - 25
            radius = 3.0 + np.sin(i * 0.1)
            f.write(f"{dist:.2f} {radius:.2f}\n")
    
    # Write mock SPH file (simplified PDB format)
    with open(sph_file, 'w') as f:
        for i in range(100):
            x, y, z = i * 0.5 - 25, 0.0, 0.0
            radius = 3.0 + np.sin(i * 0.1)
            f.write(f"ATOM  {i+1:5d}  SPH SPH A{i+1:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{radius:6.2f}\n")
    
    return {"plt": plt_file, "sph": sph_file}


# ============================================================================
# Synthetic Trajectory Fixtures
# ============================================================================

@pytest.fixture
def correlated_trajectory():
    """Create a trajectory with known correlations."""
    n_atoms = 100
    n_frames = 500
    
    # Create correlated motion
    t = np.linspace(0, 10*np.pi, n_frames)
    
    # Group 1: atoms 0-24 move together
    group1 = np.outer(np.sin(t), np.ones(25))
    
    # Group 2: atoms 25-49 move opposite to group 1
    group2 = np.outer(-np.sin(t), np.ones(25))
    
    # Group 3: atoms 50-74 move independently
    group3 = np.random.randn(n_frames, 25)
    
    # Group 4: atoms 75-99 follow cosine
    group4 = np.outer(np.cos(t), np.ones(25))
    
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[:, :25, 0] = group1
    positions[:, 25:50, 0] = group2
    positions[:, 50:75, 0] = group3
    positions[:, 75:, 0] = group4
    
    # Add y and z components
    positions[:, :, 1] = np.random.randn(n_frames, n_atoms) * 0.1
    positions[:, :, 2] = np.random.randn(n_frames, n_atoms) * 0.1
    
    return positions


@pytest.fixture
def pca_test_data():
    """Create data with known principal components."""
    n_samples = 1000
    n_features = 50
    
    # Create data with 3 main components
    components = np.random.randn(3, n_features)
    components = components / np.linalg.norm(components, axis=1)[:, np.newaxis]
    
    # Generate data
    scores = np.random.randn(n_samples, 3) * np.array([10, 5, 2])
    noise = np.random.randn(n_samples, n_features) * 0.1
    
    data = scores @ components + noise
    
    return {
        "data": data,
        "true_components": components,
        "true_variance": np.array([100, 25, 4])
    }


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def assert_array_almost_equal():
    """Helper to assert arrays are almost equal."""
    def _assert(actual, expected, decimal=7, err_msg=''):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=decimal, err_msg=err_msg
        )
    return _assert


@pytest.fixture
def assert_allclose():
    """Helper to assert arrays are close."""
    def _assert(actual, expected, rtol=1e-7, atol=0, err_msg=''):
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, err_msg=err_msg
        )
    return _assert


# ============================================================================
# Mock Objects
# ============================================================================

@pytest.fixture
def mock_matplotlib(monkeypatch):
    """Mock matplotlib for headless testing."""
    import matplotlib
    monkeypatch.setattr(matplotlib, 'use', lambda x: None)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)
    return plt


@pytest.fixture
def mock_hole_executable(monkeypatch, temp_dir):
    """Mock HOLE executable for testing without installation."""
    mock_exe = temp_dir / "hole"
    mock_exe.touch()
    mock_exe.chmod(0o755)
    
    # Create a simple shell script that generates output
    with open(mock_exe, 'w') as f:
        f.write("#!/bin/sh\n")
        f.write("echo 'HOLE mock output'\n")
    
    return str(mock_exe)


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def benchmark_universe():
    """Create a large universe for benchmarking."""
    n_atoms = 10000
    n_frames = 100
    
    u = mda.Universe.empty(n_atoms, trajectory=True)
    
    # Add frames
    for _ in range(n_frames):
        u.trajectory.add_frame()
        u.atoms.positions = np.random.randn(n_atoms, 3) * 50
    
    return u


@pytest.fixture
def timer():
    """Simple timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = {}
        
        def start(self, name):
            self.times[name] = time.time()
        
        def stop(self, name):
            if name in self.times:
                elapsed = time.time() - self.times[name]
                del self.times[name]
                return elapsed
            return None
    
    return Timer()


# ============================================================================
# Validation Data Fixtures
# ============================================================================

@pytest.fixture
def reference_rmsd():
    """Reference RMSD values for validation."""
    return {
        "ca_rmsd": np.array([0.0, 1.2, 2.3, 3.1, 2.8]),
        "backbone_rmsd": np.array([0.0, 1.5, 2.8, 3.5, 3.2])
    }


@pytest.fixture
def reference_contacts():
    """Reference contact data for validation."""
    return {
        "native_contacts": 0.85,
        "hydrogen_bonds": 42,
        "salt_bridges": 8
    }


@pytest.fixture
def reference_free_energy():
    """Reference free energy values."""
    return {
        "fep": -3.2,  # kcal/mol
        "fep_error": 0.3,
        "ti": -3.1,
        "ti_error": 0.2,
        "bar": -3.15,
        "bar_error": 0.25
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_hole: marks tests that require HOLE installation"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )