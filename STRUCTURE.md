# MDToolkit Repository Structure

```
MDToolkit/
│
├── 📄 README.md                    # Comprehensive project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 pyproject.toml              # Modern Python packaging configuration
├── 📄 setup.py                    # Package setup script
├── 📄 requirements.txt            # Core dependencies
├── 📄 environment.yml             # Conda environment specification
├── 📄 .gitignore                  # Git ignore patterns
│
├── 📁 mdtoolkit/                  # Main package directory
│   ├── __init__.py               # Package initialization
│   │
│   ├── 📁 core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── trajectory.py         # Unified trajectory handler
│   │   ├── selections.py         # Atom selection utilities
│   │   └── utils.py              # General utilities
│   │
│   ├── 📁 structure/             # Structural analysis
│   │   ├── __init__.py
│   │   ├── rmsd.py              # RMSD/RMSF calculations
│   │   ├── contacts.py          # Contact analysis (H-bonds, salt bridges)
│   │   ├── distances.py         # Distance calculations
│   │   ├── sasa.py              # Solvent accessible surface area
│   │   └── geometry.py          # Geometric calculations
│   │
│   ├── 📁 dynamics/              # Dynamics analysis (INTEGRATED)
│   │   ├── __init__.py
│   │   ├── correlation.py       # Correlation analysis
│   │   ├── pca.py              # PCA with sklearn validation
│   │   └── covariance.py        # Covariance calculations
│   │
│   ├── 📁 visualization/         # Plotting and visualization
│   │   ├── __init__.py
│   │   ├── plots.py            # Main plotting functions
│   │   ├── plot_templates.py   # Publication templates
│   │   └── colors.py           # Color schemes
│   │
│   ├── 📁 workflows/            # Complete analysis pipelines
│   │   ├── __init__.py
│   │   ├── standard_analysis.py # Standard MD analysis pipeline
│   │   └── binding_analysis.py  # Protein-ligand analysis
│   │
│   └── 📁 legacy/               # Legacy scripts for reference
│       ├── vmd_scripts/         # Essential VMD scripts
│       └── matlab_scripts/      # MATLAB functions (deprecated)
│
├── 📁 notebooks/                 # Jupyter notebook examples
│   ├── 01_integrated_analysis_workflow.ipynb
│   └── examples/
│       └── data/                # Sample data directory
│
├── 📁 tests/                    # Unit tests
│   ├── test_structure.py       # Structure module tests
│   └── test_dynamics.py        # Dynamics module tests
│
├── 📁 docs/                     # Documentation
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Documentation index
│   └── source/                 # Documentation source files
│
├── 📁 scripts/                  # Standalone scripts
│   └── quick_analysis.py       # Command-line analysis tool
│
└── 📁 .github/                  # GitHub configuration
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md       # Bug report template
    │   └── feature_request.md  # Feature request template
    └── workflows/
        └── tests.yml            # CI/CD testing workflow
```

## Key Features

### 🎯 Unified Architecture
- **Single Framework**: All analysis in Python/MDAnalysis
- **No Tool Jumping**: Eliminated need for VMD/MATLAB switching
- **Integrated Modules**: PCA and correlation in single `dynamics/` module

### 📊 Analysis Capabilities
- **Structure**: RMSD, contacts, distances, SASA, geometry
- **Dynamics**: Integrated PCA/correlation with validation
- **Workflows**: Complete pipelines from trajectory to publication

### 🔧 Modern Development
- **Type Hints**: Throughout the codebase
- **Testing**: Comprehensive pytest suite
- **CI/CD**: GitHub Actions for automated testing
- **Documentation**: Sphinx with RTD theme

### 📦 Installation
```bash
conda env create -f environment.yml
conda activate mdtoolkit
pip install -e .
```

### 🚀 Quick Usage
```python
from mdtoolkit.workflows import StandardAnalysis

pipeline = StandardAnalysis("protein.pdb", "trajectory.xtc")
pipeline.run_all_analyses()  # Complete integrated analysis
```

## Migration Notes

### From Old Structure
- `correlation/` + `mkpy/` → `mdtoolkit/dynamics/` (integrated)
- `matplotlib/` → `mdtoolkit/visualization/`
- `mkvmd/` → `mdtoolkit/structure/` (Python-based)
- VMD scripts → `legacy/vmd_scripts/` (reference only)

### Key Improvements
1. **Unified correlation/PCA** in single module
2. **MDAnalysis-based** everything (no VMD dependency)
3. **sklearn validation** built into PCA
4. **Workflow pipelines** for complete analyses
5. **Publication templates** for consistent figures