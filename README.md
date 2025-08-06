# MDToolkit

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MDAnalysis](https://img.shields.io/badge/MDAnalysis-2.0%2B-orange)](https://www.mdanalysis.org/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://github.com/skblnw/MDToolkit/wiki)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**MDToolkit** is a comprehensive, integrated molecular dynamics analysis framework that unifies structural analysis, correlation dynamics, and publication-ready visualization into a single cohesive pipeline.

## Key Features

### Unified Analysis Pipeline
- **No more tool jumping** - Single Python framework eliminates switching between VMD, MATLAB, and disparate scripts
- **Integrated workflows** - Seamlessly combine RMSD, contacts, PCA, and correlation analyses
- **MDAnalysis-powered** - Built on the robust MDAnalysis ecosystem for consistency and reliability

### Comprehensive Analysis Modules
- **Structure**: RMSD/RMSF, contacts (native/H-bonds/salt bridges), distances, SASA, geometry
- **Dynamics**: Integrated PCA & correlation analysis with sklearn validation
- **Workflows**: Pre-built pipelines for standard and binding analyses
- **Visualization**: Publication-ready plots with consistent styling

### Modern Development
- **Type hints** throughout for better IDE support
- **Validation** built-in (PCA validated with sklearn)
- **Scalable** - Handles large trajectories with chunking
- **Configurable** - YAML-based configuration for reproducibility

## Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/skblnw/MDToolkit.git
cd MDToolkit

# Create conda environment
conda env create -f environment.yml
conda activate mdtoolkit

# Install in development mode
pip install -e .
```

### Manual Installation
```bash
# Create a new environment
conda create -n mdtoolkit python=3.10
conda activate mdtoolkit

# Install dependencies
pip install -r requirements.txt

# Install MDToolkit
pip install -e .
```

### Optional Dependencies
```bash
# For Markov State Models
pip install pyemma>=2.5

# For interactive visualization
pip install nglview>=3.0

# For development
pip install pytest black flake8 jupyter
```

## Quick Start

### Basic Usage
```python
from mdtoolkit.core import TrajectoryHandler
from mdtoolkit.structure import RMSDAnalysis
from mdtoolkit.dynamics import PCAAnalysis
from mdtoolkit.visualization import plot_rmsd, plot_pca

# Load trajectory
traj = TrajectoryHandler("topology.pdb", "trajectory.xtc")
traj.align_trajectory()

# RMSD analysis
rmsd = RMSDAnalysis(traj)
results = rmsd.run()
plot_rmsd(results['time'], results['rmsd'])

# PCA with validation
pca = PCAAnalysis(traj)
pca.run_mda_pca()
pca.run_sklearn_pca()  # Automatic validation
validation = pca.validate_pca()
```

### Complete Pipeline
```python
from mdtoolkit.workflows import StandardAnalysis

# Run complete analysis pipeline
pipeline = StandardAnalysis(
    topology="protein.pdb",
    trajectory="trajectory.xtc",
    output_dir="results"
)

# Executes: alignment → RMSD → contacts → PCA → correlation → visualization
results = pipeline.run_all_analyses()
pipeline.generate_report("analysis_report.html")
```

## Example Analyses

### 1. Integrated Structure-Dynamics Analysis
```python
from mdtoolkit.workflows import StandardAnalysis

pipeline = StandardAnalysis("protein.pdb", "trajectory.xtc")
pipeline.run_all_analyses()

# Access integrated results
rmsd = pipeline.results['rmsd']
pca = pipeline.results['pca']
correlation = pipeline.results['correlation']
```

### 2. Protein-Ligand Binding Analysis
```python
from mdtoolkit.workflows import BindingAnalysis

binding = BindingAnalysis(
    topology="complex.pdb",
    trajectory="trajectory.xtc",
    protein_selection="protein",
    ligand_selection="resname LIG"
)

binding.run_complete_analysis()
# Calculates: ligand RMSD, contacts, residence time, binding energy estimate
```

### 3. Correlation-PCA Integration
```python
from mdtoolkit.dynamics import CorrelationAnalysis, PCAAnalysis

# Unified correlation and PCA
corr = CorrelationAnalysis(traj)
corr_matrix = corr.calculate_correlation_matrix()
res_corr = corr.calculate_residue_correlation()

# PCA with built-in validation
pca = PCAAnalysis(traj)
mda_results = pca.run_mda_pca()
sklearn_results = pca.run_sklearn_pca()
validation = pca.validate_pca()  # Ensures consistency
```

## Repository Structure

```
MDToolkit/
├── mdtoolkit/              # Main package
│   ├── core/              # Trajectory handling and utilities
│   ├── structure/         # Structural analysis (RMSD, contacts, etc.)
│   ├── dynamics/          # PCA and correlation (integrated)
│   ├── visualization/     # Plotting and visualization
│   └── workflows/         # Complete analysis pipelines
├── notebooks/             # Jupyter notebook examples
│   ├── 01_integrated_analysis_workflow.ipynb
│   ├── 02_correlation_dynamics.ipynb
│   └── examples/         # Example data
├── tests/                # Unit tests
├── docs/                 # Documentation
├── scripts/              # Standalone scripts
└── legacy/              # VMD/MATLAB scripts for reference
```

## Documentation

### Notebooks
Interactive Jupyter notebooks demonstrate complete workflows:

1. **[Integrated Analysis](notebooks/01_integrated_analysis_workflow.ipynb)** - Complete pipeline from trajectory to publication figures
2. **[Correlation Dynamics](notebooks/02_correlation_dynamics.ipynb)** - Advanced correlation and PCA analysis
3. **[Contact Analysis](notebooks/03_contact_analysis.ipynb)** - Detailed interaction analysis
4. **[Binding Analysis](notebooks/04_binding_analysis.ipynb)** - Protein-ligand binding workflows

### API Reference
Comprehensive API documentation is available in the [docs](docs/) directory and online at [Read the Docs](https://mdtoolkit.readthedocs.io/).

## Migration Guide

### From Scattered Scripts
```python
# Old approach (multiple tools)
# VMD: mkvmd_rmsd.sh
# Python: various correlation scripts
# MATLAB: plotting

# New approach (unified)
from mdtoolkit.workflows import StandardAnalysis
pipeline = StandardAnalysis("topology.pdb", "trajectory.xtc")
pipeline.run_all_analyses()  # Everything integrated!
```

### From VMD Scripts
```python
# Old: ./mkvmd_countHbonds.sh trajectory.dcd protein.psf
# New:
from mdtoolkit.structure import HydrogenBonds
hbonds = HydrogenBonds(traj)
results = hbonds.run()
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mdtoolkit

# Run specific test module
pytest tests/test_structure.py
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use MDToolkit in your research, please cite:

```bibtex
@software{mdtoolkit2024,
  title = {MDToolkit: Integrated Molecular Dynamics Analysis Framework},
  author = {skblnw},
  year = {2024},
  url = {https://github.com/skblnw/MDToolkit},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

MDToolkit integrates and builds upon these excellent projects:
- [MDAnalysis](https://www.mdanalysis.org/) - Core trajectory analysis
- [PyEMMA](http://emma-project.org/) - Markov State Models
- [scikit-learn](https://scikit-learn.org/) - Machine learning validation
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) - Visualization

## Support

- **Issues**: [GitHub Issues](https://github.com/skblnw/MDToolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/skblnw/MDToolkit/discussions)
- **Documentation**: [Wiki](https://github.com/skblnw/MDToolkit/wiki)
- **Email**: contact@skblnw.dev

## Roadmap

- [ ] GPU acceleration for large systems
- [ ] Interactive web dashboard
- [ ] Machine learning integration
- [ ] Enhanced MSM workflows
- [ ] Automated report generation
- [ ] Cloud deployment support

---

**Made for the MD community**