#!/usr/bin/env python
"""Quick analysis script for MD trajectories."""

import argparse
import sys
from pathlib import Path

from mdtoolkit.workflows import StandardAnalysis


def main():
    """Run quick standard analysis on trajectory."""
    parser = argparse.ArgumentParser(
        description="Quick MD trajectory analysis using MDToolkit"
    )
    
    parser.add_argument(
        "topology",
        help="Topology file (PDB, GRO, PSF, etc.)"
    )
    
    parser.add_argument(
        "trajectory",
        nargs="+",
        help="Trajectory file(s) (XTC, TRR, DCD, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="mdtoolkit_analysis",
        help="Output directory (default: mdtoolkit_analysis)"
    )
    
    parser.add_argument(
        "-s", "--selection",
        default="protein and name CA",
        help="Selection for analysis (default: 'protein and name CA')"
    )
    
    parser.add_argument(
        "--rmsd-selections",
        nargs="+",
        default=None,
        help="Additional RMSD selections"
    )
    
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA analysis"
    )
    
    parser.add_argument(
        "--skip-contacts",
        action="store_true",
        help="Skip contact analysis"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    topology = Path(args.topology)
    if not topology.exists():
        print(f"Error: Topology file not found: {topology}")
        sys.exit(1)
    
    trajectories = []
    for traj in args.trajectory:
        traj_path = Path(traj)
        if not traj_path.exists():
            print(f"Error: Trajectory file not found: {traj_path}")
            sys.exit(1)
        trajectories.append(str(traj_path))
    
    # Setup configuration
    config = {
        'align_selection': args.selection,
        'pca_selection': args.selection,
        'native_contacts': not args.skip_contacts,
        'hydrogen_bonds': not args.skip_contacts,
    }
    
    if args.rmsd_selections:
        rmsd_selections = {}
        for i, sel in enumerate(args.rmsd_selections):
            rmsd_selections[f"selection_{i+1}"] = sel
        config['rmsd_selections'] = rmsd_selections
    
    # Run analysis
    print(f"Starting MDToolkit analysis...")
    print(f"  Topology: {topology}")
    print(f"  Trajectory: {trajectories}")
    print(f"  Output: {args.output}")
    
    try:
        pipeline = StandardAnalysis(
            topology=str(topology),
            trajectory=trajectories if len(trajectories) > 1 else trajectories[0],
            output_dir=args.output,
            config=config
        )
        
        # Run analyses
        print("\nRunning RMSD analysis...")
        pipeline.run_rmsd_analysis()
        
        print("Running RMSF analysis...")
        pipeline.run_rmsf_analysis()
        
        if not args.skip_contacts:
            print("Running contact analysis...")
            pipeline.run_contact_analysis()
        
        if not args.skip_pca:
            print("Running PCA analysis...")
            pipeline.run_pca_analysis()
            
            print("Running correlation analysis...")
            pipeline.run_correlation_analysis()
        
        # Generate report
        if args.report:
            print("\nGenerating HTML report...")
            pipeline.generate_report()
        
        print(f"\n✅ Analysis complete! Results saved to {args.output}/")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()