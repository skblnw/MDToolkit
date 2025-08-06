"""Binding analysis workflow for protein-ligand systems."""

import logging
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances

from ..core import TrajectoryHandler
from ..structure import ContactAnalysis, HydrogenBonds, calculate_rmsd
from ..dynamics import PCAAnalysis
from ..visualization import plot_timeseries, plot_contacts

logger = logging.getLogger(__name__)


class BindingAnalysis:
    """
    Analysis pipeline for protein-ligand binding.
    
    Includes:
    - Ligand RMSD
    - Protein-ligand contacts
    - Binding site analysis
    - Residence time calculations
    - Free energy estimates
    """
    
    def __init__(
        self,
        topology: Union[str, Path],
        trajectory: Union[str, Path, List[str]],
        protein_selection: str = "protein",
        ligand_selection: str = "resname LIG",
        binding_site_selection: Optional[str] = None,
        output_dir: Union[str, Path] = "binding_analysis"
    ):
        """
        Initialize binding analysis.
        
        Parameters
        ----------
        topology : str or Path
            Topology file
        trajectory : str, Path, or list
            Trajectory file(s)
        protein_selection : str
            Protein selection string
        ligand_selection : str
            Ligand selection string
        binding_site_selection : str, optional
            Binding site selection
        output_dir : str or Path
            Output directory
        """
        self.topology = Path(topology)
        self.trajectory = trajectory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.protein_sel = protein_selection
        self.ligand_sel = ligand_selection
        self.binding_site_sel = binding_site_selection
        
        # Load trajectory
        self.traj = TrajectoryHandler(topology, trajectory)
        self.universe = self.traj.universe
        
        # Get selections
        self.protein = self.universe.select_atoms(protein_selection)
        self.ligand = self.universe.select_atoms(ligand_selection)
        
        if binding_site_selection:
            self.binding_site = self.universe.select_atoms(binding_site_selection)
        else:
            # Auto-detect binding site (residues within 5Å of ligand in first frame)
            self.universe.trajectory[0]
            around_lig = self.universe.select_atoms(
                f"({protein_selection}) and around 5.0 ({ligand_selection})"
            )
            self.binding_site = around_lig
            self.binding_site_sel = f"resid {' '.join(map(str, np.unique(around_lig.resids)))}"
        
        self.results = {}
        
    def calculate_ligand_rmsd(
        self,
        align_protein: bool = True,
        align_selection: str = "protein and name CA"
    ) -> np.ndarray:
        """
        Calculate ligand RMSD.
        
        Parameters
        ----------
        align_protein : bool
            Whether to align protein first
        align_selection : str
            Selection for alignment
            
        Returns
        -------
        np.ndarray
            Ligand RMSD values
        """
        logger.info("Calculating ligand RMSD...")
        
        if align_protein:
            self.traj.align_trajectory(selection=align_selection)
        
        # Calculate ligand RMSD
        rmsd = calculate_rmsd(
            self.universe,
            selection=self.ligand_sel
        )
        
        # Save results
        np.savetxt(
            self.output_dir / "ligand_rmsd.txt",
            rmsd,
            header="Frame Time RMSD"
        )
        
        self.results['ligand_rmsd'] = rmsd
        return rmsd
    
    def analyze_protein_ligand_contacts(
        self,
        distance_cutoff: float = 4.0
    ) -> Dict:
        """
        Analyze protein-ligand contacts.
        
        Parameters
        ----------
        distance_cutoff : float
            Distance cutoff for contacts
            
        Returns
        -------
        dict
            Contact analysis results
        """
        logger.info("Analyzing protein-ligand contacts...")
        
        # General contacts
        contacts = ContactAnalysis(
            trajectory=self.universe,
            selection1=self.binding_site_sel,
            selection2=self.ligand_sel,
            cutoff=distance_cutoff
        )
        
        contact_results = contacts.run(detailed=True)
        
        # Hydrogen bonds
        hbonds = HydrogenBonds(
            trajectory=self.universe,
            donors=f"({self.protein_sel}) and (name N* or name O*)",
            acceptors=f"({self.ligand_sel}) and (name N* or name O*)"
        )
        
        hb_results = hbonds.run()
        
        # Identify key interacting residues
        contact_frequency = self._calculate_contact_frequency(
            contact_results['contact_pairs']
        )
        
        results = {
            'contacts': contact_results,
            'hydrogen_bonds': hb_results,
            'contact_frequency': contact_frequency,
            'persistent_contacts': contacts.get_persistent_contacts(0.5)
        }
        
        # Plot contacts
        plot_contacts(
            time=contact_results['frames'],
            contacts=contact_results['n_contacts'],
            title="Protein-Ligand Contacts",
            save_path=self.output_dir / "protein_ligand_contacts.png"
        )
        
        self.results['contacts'] = results
        return results
    
    def _calculate_contact_frequency(
        self,
        contact_pairs: List[List[Tuple]]
    ) -> pd.DataFrame:
        """Calculate contact frequency by residue."""
        residue_contacts = {}
        
        for frame_pairs in contact_pairs:
            for atom1_idx, atom2_idx in frame_pairs:
                atom1 = self.universe.atoms[atom1_idx]
                residue = atom1.residue
                key = f"{residue.resname}{residue.resid}"
                
                if key not in residue_contacts:
                    residue_contacts[key] = 0
                residue_contacts[key] += 1
        
        # Normalize by number of frames
        n_frames = len(contact_pairs)
        for key in residue_contacts:
            residue_contacts[key] /= n_frames
        
        # Convert to DataFrame
        df = pd.DataFrame(
            list(residue_contacts.items()),
            columns=['Residue', 'Contact_Frequency']
        )
        df = df.sort_values('Contact_Frequency', ascending=False)
        
        # Save
        df.to_csv(self.output_dir / "residue_contact_frequency.csv", index=False)
        
        return df
    
    def calculate_binding_site_flexibility(self) -> Dict:
        """
        Calculate binding site flexibility.
        
        Returns
        -------
        dict
            RMSF and B-factors for binding site
        """
        logger.info("Calculating binding site flexibility...")
        
        from ..structure import calculate_rmsf
        
        # Calculate RMSF for binding site
        rmsf = calculate_rmsf(
            self.universe,
            selection=self.binding_site_sel
        )
        
        # Convert to B-factors
        b_factors = 8 * np.pi**2 * rmsf**2 / 3
        
        results = {
            'rmsf': rmsf,
            'b_factors': b_factors,
            'mean_rmsf': np.mean(rmsf),
            'max_rmsf': np.max(rmsf)
        }
        
        self.results['flexibility'] = results
        return results
    
    def calculate_residence_time(
        self,
        distance_threshold: float = 5.0,
        min_residence_frames: int = 10
    ) -> Dict:
        """
        Calculate ligand residence time in binding site.
        
        Parameters
        ----------
        distance_threshold : float
            Distance to define bound state
        min_residence_frames : int
            Minimum frames for residence event
            
        Returns
        -------
        dict
            Residence time statistics
        """
        logger.info("Calculating residence time...")
        
        # Calculate center of mass distance
        com_distances = []
        
        for ts in self.universe.trajectory:
            lig_com = self.ligand.center_of_mass()
            site_com = self.binding_site.center_of_mass()
            dist = np.linalg.norm(lig_com - site_com)
            com_distances.append(dist)
        
        com_distances = np.array(com_distances)
        
        # Identify bound states
        bound = com_distances < distance_threshold
        
        # Find residence events
        events = []
        current_event = []
        
        for i, is_bound in enumerate(bound):
            if is_bound:
                current_event.append(i)
            else:
                if len(current_event) >= min_residence_frames:
                    events.append(len(current_event))
                current_event = []
        
        # Add last event if still bound
        if len(current_event) >= min_residence_frames:
            events.append(len(current_event))
        
        if events:
            residence_times = np.array(events) * self.universe.trajectory.dt
            mean_residence = np.mean(residence_times)
            max_residence = np.max(residence_times)
        else:
            residence_times = np.array([])
            mean_residence = 0
            max_residence = 0
        
        results = {
            'com_distances': com_distances,
            'bound_fraction': np.mean(bound),
            'residence_events': events,
            'residence_times': residence_times,
            'mean_residence_time': mean_residence,
            'max_residence_time': max_residence,
            'n_binding_events': len(events)
        }
        
        # Plot COM distance
        time = np.arange(len(com_distances)) * self.universe.trajectory.dt
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, com_distances, linewidth=2)
        ax.axhline(y=distance_threshold, color='r', linestyle='--', 
                  label=f'Threshold ({distance_threshold} Å)')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('COM Distance (Å)')
        ax.set_title('Ligand-Binding Site Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(self.output_dir / "com_distance.png", dpi=300)
        plt.close()
        
        self.results['residence_time'] = results
        return results
    
    def estimate_binding_free_energy(
        self,
        temperature: float = 300.0
    ) -> Dict:
        """
        Estimate binding free energy from residence time.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
            
        Returns
        -------
        dict
            Free energy estimates
        """
        if 'residence_time' not in self.results:
            self.calculate_residence_time()
        
        bound_fraction = self.results['residence_time']['bound_fraction']
        
        if bound_fraction > 0 and bound_fraction < 1:
            # Calculate free energy from population
            kT = 0.001987 * temperature  # kcal/mol
            dG = -kT * np.log(bound_fraction / (1 - bound_fraction))
            
            # Estimate from residence time if available
            mean_residence = self.results['residence_time']['mean_residence_time']
            if mean_residence > 0:
                # Rough estimate assuming diffusion-limited on-rate
                k_off = 1.0 / mean_residence
                k_on_estimate = 1e8  # M^-1 s^-1 (diffusion-limited)
                Kd_estimate = k_off / k_on_estimate
                dG_kinetic = kT * np.log(Kd_estimate)
            else:
                dG_kinetic = None
        else:
            dG = None
            dG_kinetic = None
        
        results = {
            'dG_population': dG,
            'dG_kinetic': dG_kinetic,
            'bound_fraction': bound_fraction,
            'temperature': temperature
        }
        
        self.results['binding_energy'] = results
        return results
    
    def run_complete_analysis(self) -> Dict:
        """Run complete binding analysis pipeline."""
        logger.info("Starting binding analysis pipeline...")
        
        # Run all analyses
        self.calculate_ligand_rmsd()
        self.analyze_protein_ligand_contacts()
        self.calculate_binding_site_flexibility()
        self.calculate_residence_time()
        self.estimate_binding_free_energy()
        
        # Generate summary
        self.generate_summary()
        
        logger.info(f"Binding analysis complete! Results saved to {self.output_dir}")
        
        return self.results
    
    def generate_summary(self) -> None:
        """Generate binding analysis summary."""
        summary = []
        
        summary.append("=" * 60)
        summary.append("BINDING ANALYSIS SUMMARY")
        summary.append("=" * 60)
        
        # System info
        summary.append(f"\nSystem: {self.topology.name}")
        summary.append(f"Frames analyzed: {self.traj.n_frames}")
        summary.append(f"Ligand atoms: {len(self.ligand)}")
        summary.append(f"Binding site residues: {len(self.binding_site.residues)}")
        
        # Ligand RMSD
        if 'ligand_rmsd' in self.results:
            rmsd = self.results['ligand_rmsd'][:, 2]
            summary.append(f"\nLigand RMSD:")
            summary.append(f"  Mean: {np.mean(rmsd):.2f} Å")
            summary.append(f"  Max: {np.max(rmsd):.2f} Å")
        
        # Contacts
        if 'contacts' in self.results:
            contacts = self.results['contacts']['contacts']['n_contacts']
            summary.append(f"\nProtein-Ligand Contacts:")
            summary.append(f"  Mean: {np.mean(contacts):.1f}")
            summary.append(f"  Max: {np.max(contacts)}")
            
            # Top interacting residues
            freq_df = self.results['contacts']['contact_frequency']
            summary.append(f"\nTop 5 Interacting Residues:")
            for i, row in freq_df.head(5).iterrows():
                summary.append(f"  {row['Residue']}: {row['Contact_Frequency']:.2%}")
        
        # Residence time
        if 'residence_time' in self.results:
            rt = self.results['residence_time']
            summary.append(f"\nResidence Time:")
            summary.append(f"  Bound fraction: {rt['bound_fraction']:.2%}")
            summary.append(f"  Number of binding events: {rt['n_binding_events']}")
            if rt['mean_residence_time'] > 0:
                summary.append(f"  Mean residence time: {rt['mean_residence_time']:.1f} ps")
        
        # Binding energy
        if 'binding_energy' in self.results:
            be = self.results['binding_energy']
            if be['dG_population'] is not None:
                summary.append(f"\nBinding Free Energy Estimate:")
                summary.append(f"  ΔG (population): {be['dG_population']:.2f} kcal/mol")
        
        # Write summary
        summary_text = "\n".join(summary)
        with open(self.output_dir / "binding_summary.txt", 'w') as f:
            f.write(summary_text)
        
        print(summary_text)