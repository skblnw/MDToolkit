# Contributing to MDToolkit

First off, thank you for considering contributing to MDToolkit! It's people like you that make MDToolkit such a great tool for the molecular dynamics community.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to mdtoolkit@yourdomain.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples** (include copy-pasteable snippets)
* **Describe the behavior you observed and what you expected**
* **Include system details** (OS, Python version, MDAnalysis version)
* **Include trajectory format** if relevant

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

* **Use a clear and descriptive title**
* **Provide a detailed description** of the suggested enhancement
* **Provide specific examples** to demonstrate the enhancement
* **Describe the current behavior** and explain how it would change
* **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Set up your development environment:
   ```bash
   conda env create -f environment.yml
   conda activate mdtoolkit
   pip install -e ".[dev]"
   ```

3. Make your changes:
   * Add or update tests as appropriate
   * Update documentation if needed
   * Follow the existing code style

4. Ensure tests pass:
   ```bash
   pytest
   pytest --cov=mdtoolkit  # Check coverage
   ```

5. Format your code:
   ```bash
   black mdtoolkit/
   flake8 mdtoolkit/
   ```

6. Commit your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```

8. Open a Pull Request

## Development Guidelines

### Code Style

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use [black](https://github.com/psf/black) for formatting
* Use type hints where possible
* Maximum line length: 100 characters
* Use descriptive variable names

### Documentation

* Add docstrings to all functions/classes (NumPy style)
* Update README.md if adding new features
* Add examples to docstrings
* Create/update notebook examples for major features

### Testing

* Write tests for new functionality
* Maintain or improve code coverage
* Use pytest for testing
* Place tests in appropriate test modules

### Commit Messages

* Use present tense ("Add feature" not "Added feature")
* Use imperative mood ("Move cursor to..." not "Moves cursor to...")
* Reference issues and pull requests liberally
* Consider starting with an emoji:
  * 🎨 `:art:` - Improving structure/format
  * ⚡ `:zap:` - Improving performance
  * 🐛 `:bug:` - Fixing a bug
  * ✨ `:sparkles:` - Adding new feature
  * 📝 `:memo:` - Writing docs
  * ♻️ `:recycle:` - Refactoring code
  * ✅ `:white_check_mark:` - Adding tests

## Project Structure

```
MDToolkit/
├── mdtoolkit/          # Main package code
│   ├── core/          # Core functionality
│   ├── structure/     # Structural analysis
│   ├── dynamics/      # Dynamics analysis
│   ├── visualization/ # Plotting functions
│   └── workflows/     # Analysis pipelines
├── tests/             # Test files
├── notebooks/         # Example notebooks
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

## Adding New Analysis Methods

When adding a new analysis method:

1. **Choose the right module**:
   * `structure/` for structural properties
   * `dynamics/` for time-dependent properties
   * `workflows/` for complete pipelines

2. **Follow the existing pattern**:
   ```python
   class YourAnalysis:
       def __init__(self, trajectory, **kwargs):
           # Initialize
       
       def run(self):
           # Main analysis
           
       def save_results(self):
           # Save outputs
   ```

3. **Add visualization**:
   * Create plotting function in `visualization/`
   * Use consistent style with existing plots

4. **Document thoroughly**:
   * Add comprehensive docstrings
   * Include usage examples
   * Update relevant notebooks

5. **Test extensively**:
   * Unit tests for individual methods
   * Integration tests for workflows
   * Test with different trajectory formats

## Questions?

Feel free to open an issue for:
* Questions about the codebase
* Clarification on how to contribute
* Discussion about potential features

## Recognition

Contributors will be acknowledged in:
* The AUTHORS file
* Release notes
* Documentation

Thank you for contributing to MDToolkit! 🎉