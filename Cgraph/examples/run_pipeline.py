#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for running the Multi-Omics Integration Pipeline.
This script demonstrates how to use the pipeline on sample data.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Cgraph.pipeline import MultiOmicsIntegration
from Cgraph.src.visualization.embedding_viz import (
    visualize_participant_embeddings,
    plot_modality_importance,
    discover_cross_modality_correlations
)
from Cgraph.src.data_processing.pseudo_data_generator import (
    generate_pseudo_omics_data,
    generate_high_accuracy_data
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(output_dir, n_participants=100, n_features=50, missing_rate=0.2, 
                      high_accuracy=False, n_clusters=3):
    """
    Create sample data for the pipeline.
    
    Args:
        output_dir (str): Directory to save the sample data
        n_participants (int): Number of participants
        n_features (int): Number of features per modality
        missing_rate (float): Rate of missing data
        high_accuracy (bool): Whether to generate high-accuracy data with clear patterns
        n_clusters (int): Number of clusters to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use advanced pseudo data generator if available
    if high_accuracy:
        logger.info(f"Generating high-accuracy pseudo data with {n_participants} participants and {n_clusters} clusters")
        metadata = generate_high_accuracy_data(
            output_dir=output_dir,
            n_participants=n_participants,
            n_clusters=n_clusters,
            cluster_separation=3.0,
            high_signal_ratio=0.4,
            random_state=42
        )
        return metadata
    else:
        logger.info(f"Generating standard pseudo data with {n_participants} participants")
        metadata = generate_pseudo_omics_data(
            output_dir=output_dir,
            n_participants=n_participants,
            n_clusters=n_clusters,
            feature_counts={
                'metabolomics': n_features,
                'proteomics': n_features,
                'biochemistry': n_features,
                'lifestyle': n_features,
                'lipidomics': n_features
            },
            missing_rates={
                'metabolomics': missing_rate,
                'proteomics': missing_rate,
                'biochemistry': missing_rate,
                'lifestyle': missing_rate * 0.5,  # Less missing data for lifestyle
                'lipidomics': missing_rate
            },
            add_correlated_features=True,
            random_state=42
        )
        return metadata


def main():
    """Run the example pipeline."""
    parser = argparse.ArgumentParser(description="Run the Multi-Omics Integration Pipeline example")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Directory with data files")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--generate_data", action="store_true", help="Generate sample data")
    parser.add_argument("--high_accuracy", action="store_true", help="Generate high-accuracy data with clear patterns")
    parser.add_argument("--n_participants", type=int, default=100, help="Number of participants for sample data")
    parser.add_argument("--n_features", type=int, default=50, help="Number of features per modality for sample data")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for sample data")
    parser.add_argument("--missing_rate", type=float, default=0.2, help="Rate of missing data")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate sample data if requested
    if args.generate_data:
        metadata = create_sample_data(
            args.data_dir, 
            n_participants=args.n_participants, 
            n_features=args.n_features,
            missing_rate=args.missing_rate,
            high_accuracy=args.high_accuracy,
            n_clusters=args.n_clusters
        )
        
        if metadata and 'cluster_assignments' in metadata:
            logger.info(f"Generated data with {len(set(metadata['cluster_assignments']))} clusters")
            if args.high_accuracy:
                logger.info("Generated high-accuracy data with enhanced signal-to-noise ratio")
                
            # Log cross-modality correlations if they exist
            if 'cross_modality_correlations' in metadata and metadata['cross_modality_correlations']:
                n_corr = len(metadata['cross_modality_correlations'])
                logger.info(f"Added {n_corr} cross-modality correlations for biological realism")
    
    # Initialize pipeline
    pipeline = MultiOmicsIntegration(
        embedding_dim=128,  # Smaller for example
        n_heads=4,
        n_layers=2,
        correlation_threshold=0.5
    )
    
    # Load data
    pipeline.load_data(
        data_dir=args.data_dir,
        metabolomics_file="metabolomics.csv",
        proteomics_file="proteomics.csv",
        biochemistry_file="biochemistry.csv",
        lifestyle_file="lifestyle.csv",
        lipidomics_file="lipidomics.csv"
    )
    
    # Run pipeline
    embeddings = pipeline.run()
    
    # Evaluate modality importance
    modality_importance = pipeline.evaluate_modality_importance()
    
    # Save results
    pipeline.save_results(args.output_dir)
    
    # Create visualizations
    visualize_participant_embeddings(
        embeddings['participant'],
        participant_ids=pipeline.participant_ids,
        method='umap',
        output_file=os.path.join(args.output_dir, 'participant_embeddings.png')
    )
    
    plot_modality_importance(
        modality_importance,
        output_file=os.path.join(args.output_dir, 'modality_importance.png')
    )
    
    # Discover cross-modality correlations
    if 'metabolomics' in embeddings and 'proteomics' in embeddings:
        # Using a lower correlation threshold (0.4) to ensure we capture more correlations
        correlations = discover_cross_modality_correlations(
            pipeline,
            source_modality='metabolomics',
            target_modality='proteomics',
            correlation_threshold=0.4,  # Lowered from 0.7 to catch more correlations
            output_file=os.path.join(args.output_dir, 'cross_modality_correlations.csv'),
            ignore_missing_features=True  # Continue even if feature names are missing
        )
        
        logger.info(f"Found {len(correlations)} significant correlations between metabolomics and proteomics")
        
        # Print top correlations
        if correlations:
            logger.info("\nTop correlations:")
            for corr in correlations[:5]:
                logger.info(f"{corr['source_feature']} ({corr['source_modality']}) <--> "
                           f"{corr['target_feature']} ({corr['target_modality']}): {corr['score']:.3f}")
        else:
            logger.warning("No significant cross-modality correlations found. You may need to lower the threshold further.")
    
    logger.info(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()