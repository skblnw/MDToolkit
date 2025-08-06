from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="md-toolkit",
    version="1.0.0",
    author="MD-Toolkit Contributors",
    description="A comprehensive molecular dynamics analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/md-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "MDAnalysis>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "msm": ["pyemma>=2.5.0"],
        "visualization": ["nglview>=3.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=3.9",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "mdtk-rmsd=scripts.quick_rmsd:main",
            "mdtk-batch=scripts.batch_analysis:main",
            "mdtk-convert=scripts.convert_trajectory:main",
        ],
    },
)