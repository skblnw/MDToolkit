MDToolkit Documentation
=======================

Welcome to MDToolkit's documentation!

MDToolkit is a comprehensive, integrated molecular dynamics analysis framework that unifies structural analysis, correlation dynamics, and publication-ready visualization into a single cohesive pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   core/index
   structure/index
   dynamics/index
   visualization/index
   workflows/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/structure
   api/dynamics
   api/visualization
   api/workflows

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   changelog
   license

Key Features
------------

* **Unified Pipeline**: No more switching between VMD, MATLAB, and Python scripts
* **Integrated Analysis**: Seamless combination of RMSD, contacts, PCA, and correlation
* **Validation Built-in**: Automatic PCA validation with sklearn
* **Publication Ready**: Consistent, professional visualization templates
* **Modern Python**: Type hints, logging, and modular design

Quick Example
-------------

.. code-block:: python

   from mdtoolkit.workflows import StandardAnalysis

   # Run complete analysis pipeline
   pipeline = StandardAnalysis(
       topology="protein.pdb",
       trajectory="trajectory.xtc"
   )
   
   # Automatically performs:
   # - Trajectory alignment
   # - RMSD/RMSF analysis
   # - Contact analysis
   # - PCA with validation
   # - Correlation analysis
   # - Visualization
   results = pipeline.run_all_analyses()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`