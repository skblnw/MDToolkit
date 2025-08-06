"""Standard analysis pipeline for MD trajectories."""

import logging
from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda

from ..core import TrajectoryHandler
from ..structure import (
    RMSDAnalysis, 
    ContactAnalysis, 
    NativeContacts,
    HydrogenBonds,
    calculate_rmsf
)
from ..dynamics import (
    CorrelationAnalysis,
    PCAAnalysis
)
from ..visualization import (
    plot_rmsd,
    plot_rmsf,
    plot_correlation_matrix,
    plot_pca,
    plot_free_energy_landscape
)

logger = logging.getLogger(__name__)


class StandardAnalysis:
    """
    Complete standard analysis pipeline for MD trajectories.
    
    This workflow integrates:
    - Trajectory loading and alignment
    - RMSD/RMSF calculations
    - Contact analysis
    - PCA and correlation analysis
    - Automated visualization
    - Report generation
    """
    
    def __init__(
        self,
        topology: Union[str, Path],
        trajectory: Union[str, Path, List[str]],
        output_dir: Union[str, Path] = "analysis_output",
        config: Optional[Dict] = None
    ):
        """
        Initialize standard analysis pipeline.
        
        Parameters
        ----------
        topology : str or Path
            Topology file path
        trajectory : str, Path, or list
            Trajectory file(s)
        output_dir : str or Path
            Output directory for results
        config : dict, optional
            Configuration dictionary
        """
        self.topology = Path(topology)
        self.trajectory = trajectory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'align_selection': 'protein and name CA',
            'rmsd_selections': {
                'backbone': 'protein and backbone',
                'ca': 'protein and name CA'
            },
            'pca_selection': 'protein and name CA',
            'contact_cutoff': 5.0,
            'hbond_distance': 3.5,
            'hbond_angle': 150.0,
            'n_pca_components': 10,
            'validate_pca': True
        }
        
        if config:
            self.config.update(config)
        
        # Load trajectory
        self.trajectory_handler = TrajectoryHandler(
            topology=self.topology,
            trajectory=self.trajectory
        )
        
        # Align if requested
        if self.config.get('align', True):
            self.trajectory_handler.align_trajectory(
                selection=self.config['align_selection']
            )
            
        self.results = {}
        
    def run_rmsd_analysis(self) -> Dict:
        """Run RMSD analysis."""
        logger.info("Running RMSD analysis...")
        
        rmsd_analysis = RMSDAnalysis(
            trajectory=self.trajectory_handler,
            align_selection=self.config['align_selection'],
            analysis_selections=self.config['rmsd_selections']
        )
        
        rmsd_results = rmsd_analysis.run()
        rmsd_analysis.save_results(self.output_dir / "rmsd")
        
        # Plot RMSD
        for name, data in rmsd_results.items():
            plot_rmsd(
                time=data['time'] / 1000,  # Convert to ns
                rmsd=data['rmsd'],
                title=f"RMSD - {name}",
                save_path=self.output_dir / f"rmsd_{name}.png"
            )
        
        self.results['rmsd'] = rmsd_results
        return rmsd_results
    
    def run_rmsf_analysis(self) -> np.ndarray:
        """Run RMSF analysis."""
        logger.info("Running RMSF analysis...")
        
        selection = self.config.get('rmsf_selection', 'protein and name CA')
        atoms = self.trajectory_handler.universe.select_atoms(selection)
        
        rmsf = calculate_rmsf(
            self.trajectory_handler.universe,
            selection=selection
        )
        
        # Save results
        np.savetxt(
            self.output_dir / "rmsf.txt",
            np.column_stack([atoms.resids, rmsf]),
            header="Residue RMSF"
        )
        
        # Plot RMSF
        plot_rmsf(
            residues=atoms.resids,
            rmsf=rmsf,
            save_path=self.output_dir / "rmsf.png"
        )
        
        self.results['rmsf'] = rmsf
        return rmsf
    
    def run_contact_analysis(self) -> Dict:
        """Run contact analysis."""
        logger.info("Running contact analysis...")
        
        results = {}
        
        # Native contacts
        if self.config.get('native_contacts', True):
            nc = NativeContacts(
                trajectory=self.trajectory_handler,
                selection=self.config['pca_selection']
            )
            nc_results = nc.run()
            results['native_contacts'] = nc_results
            
        # Hydrogen bonds
        if self.config.get('hydrogen_bonds', True):
            hb = HydrogenBonds(
                trajectory=self.trajectory_handler,
                distance_cutoff=self.config['hbond_distance'],
                angle_cutoff=self.config['hbond_angle']
            )
            hb_results = hb.run()
            results['hydrogen_bonds'] = hb_results
            
            # Get persistent H-bonds
            persistent = hb.get_persistent_hbonds(persistence_cutoff=0.5)
            results['persistent_hbonds'] = persistent
        
        self.results['contacts'] = results
        return results
    
    def run_pca_analysis(self) -> Dict:
        """Run PCA analysis with optional validation."""
        logger.info("Running PCA analysis...")
        
        pca = PCAAnalysis(
            trajectory=self.trajectory_handler,
            selection=self.config['pca_selection'],
            align=False  # Already aligned
        )
        
        # Run MDAnalysis PCA
        mda_results = pca.run_mda_pca()
        
        # Validate with sklearn if requested
        if self.config['validate_pca']:
            sklearn_results = pca.run_sklearn_pca(
                n_components=self.config['n_pca_components']
            )
            validation = pca.validate_pca()
            logger.info(f"PCA validation: {validation}")
        
        # Calculate cosine content
        cosine_content = pca.calculate_cosine_content(n_components=3)
        
        # Save results
        pca.save_results(self.output_dir / "pca")
        
        # Plot PCA
        projections = mda_results['transformed']
        
        # 2D projection
        plot_pca(
            projections=projections,
            pc_x=0,
            pc_y=1,
            save_path=self.output_dir / "pca_projection.png"
        )
        
        # Free energy landscape
        plot_free_energy_landscape(
            x=projections[:, 0],
            y=projections[:, 1],
            save_path=self.output_dir / "free_energy.png"
        )
        
        results = {
            'pca': mda_results,
            'cosine_content': cosine_content
        }
        
        if self.config['validate_pca']:
            results['validation'] = validation
        
        self.results['pca'] = results
        return results
    
    def run_correlation_analysis(self) -> Dict:
        """Run correlation analysis."""
        logger.info("Running correlation analysis...")
        
        corr = CorrelationAnalysis(
            trajectory=self.trajectory_handler,
            selection=self.config['pca_selection'],
            align=False  # Already aligned
        )
        
        # Extract positions and calculate correlation
        positions = corr.extract_positions()
        corr_matrix = corr.calculate_correlation_matrix(method="pearson")
        
        # Residue-level correlation
        res_corr = corr.calculate_residue_correlation()
        
        # Save results
        corr.save_results(self.output_dir / "correlation")
        
        # Plot correlation matrices
        plot_correlation_matrix(
            correlation=corr_matrix,
            title="Atom Correlation",
            save_path=self.output_dir / "correlation_atom.png"
        )
        
        plot_correlation_matrix(
            correlation=res_corr,
            title="Residue Correlation",
            save_path=self.output_dir / "correlation_residue.png"
        )
        
        results = {
            'atom_correlation': corr_matrix,
            'residue_correlation': res_corr
        }
        
        self.results['correlation'] = results
        return results
    
    def run_all_analyses(self) -> Dict:
        """Run all analyses in the pipeline."""
        logger.info("Starting complete analysis pipeline...")
        
        # Run all analyses
        self.run_rmsd_analysis()
        self.run_rmsf_analysis()
        self.run_contact_analysis()
        self.run_pca_analysis()
        self.run_correlation_analysis()
        
        # Create summary
        self.create_summary()
        
        logger.info(f"Analysis complete! Results saved to {self.output_dir}")
        
        return self.results
    
    def create_summary(self) -> pd.DataFrame:
        """Create summary DataFrame of all analyses."""
        summary_data = {}
        
        # Add time
        if 'rmsd' in self.results:
            time_data = self.results['rmsd']['ca']['time'] / 1000  # ns
            summary_data['Time (ns)'] = time_data
            
            # Add RMSD
            for name, data in self.results['rmsd'].items():
                summary_data[f'RMSD_{name} (Å)'] = data['rmsd']
        
        # Add contacts
        if 'contacts' in self.results:
            if 'native_contacts' in self.results['contacts']:
                summary_data['Native_Contacts_Q'] = \
                    self.results['contacts']['native_contacts']['q']
            
            if 'hydrogen_bonds' in self.results['contacts']:
                summary_data['H_bonds'] = \
                    self.results['contacts']['hydrogen_bonds']['n_hbonds']
        
        # Add PCA projections
        if 'pca' in self.results:
            for i in range(min(3, self.results['pca']['pca']['n_components'])):
                summary_data[f'PC{i+1}'] = \
                    self.results['pca']['pca']['transformed'][:, i]
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save summary
        df.to_csv(self.output_dir / "analysis_summary.csv", index=False)
        
        # Calculate correlations
        correlations = df.corr()
        correlations.to_csv(self.output_dir / "metric_correlations.csv")
        
        self.results['summary'] = df
        return df
    
    def generate_report(self, filename: str = "report.html") -> None:
        """
        Generate HTML report of analysis results.
        
        Parameters
        ----------
        filename : str
            Output filename for report
        """
        report_path = self.output_dir / filename
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MD Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
                img {{ max-width: 800px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>MD Trajectory Analysis Report</h1>
            
            <h2>System Information</h2>
            <div class="metric">
                <p>Topology: {self.topology.name}</p>
                <p>Number of frames: {self.trajectory_handler.n_frames}</p>
                <p>Number of atoms: {self.trajectory_handler.n_atoms}</p>
            </div>
            
            <h2>RMSD Analysis</h2>
            <img src="rmsd_ca.png" alt="RMSD CA">
            
            <h2>RMSF Analysis</h2>
            <img src="rmsf.png" alt="RMSF">
            
            <h2>PCA Analysis</h2>
            <img src="pca_projection.png" alt="PCA Projection">
            <img src="free_energy.png" alt="Free Energy Landscape">
        """
        
        if 'pca' in self.results and 'cosine_content' in self.results['pca']:
            cc = self.results['pca']['cosine_content']
            html_content += f"""
            <div class="metric">
                <p>Cosine content (PC1-3): {cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f}</p>
            </div>
            """
        
        html_content += """
            <h2>Correlation Analysis</h2>
            <img src="correlation_residue.png" alt="Residue Correlation">
            
            <h2>Summary Statistics</h2>
        """
        
        if 'summary' in self.results:
            df = self.results['summary']
            html_content += df.describe().to_html()
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_path}")