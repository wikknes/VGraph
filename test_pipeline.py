#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Cgraph pipeline using the pseudo data generator.
This runs a complete end-to-end test of the pipeline.
"""

import os
import logging
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

from Cgraph.pipeline import MultiOmicsIntegration
from Cgraph.src.data_processing.pseudo_data_generator import generate_high_accuracy_data
from Cgraph.src.visualization.embedding_viz import (
    visualize_participant_embeddings,
    plot_modality_importance,
    discover_cross_modality_correlations
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test_pipeline(
    data_dir="test_data",
    output_dir="test_results",
    n_participants=100,
    n_clusters=3,
    embedding_dim=128,
    n_heads=4,
    n_layers=2
):
    """
    Run a complete test of the Cgraph pipeline using generated pseudo data.
    
    Args:
        data_dir (str): Directory to save generated data
        output_dir (str): Directory to save results
        n_participants (int): Number of participants to generate
        n_clusters (int): Number of clusters in the data
        embedding_dim (int): Dimension of embeddings
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        
    Returns:
        bool: True if test completes successfully
    """
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Generate high-accuracy test data
        logger.info("Generating high-accuracy test data...")
        metadata = generate_high_accuracy_data(
            output_dir=data_dir,
            n_participants=n_participants,
            n_clusters=n_clusters,
            cluster_separation=3.0,
            high_signal_ratio=0.4,
            random_state=42
        )
        
        logger.info(f"Generated {n_participants} participants with {n_clusters} clusters")
        
        # Step 2: Initialize the pipeline
        logger.info("Initializing pipeline...")
        pipeline = MultiOmicsIntegration(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            correlation_threshold=0.5,
            n_neighbors=15,
            use_mini_batch=True,
            batch_size=32,
            use_amp=False,  # Set to True for GPU acceleration
            early_stopping=True,
            patience=10
        )
        
        # Step 3: Load data
        logger.info("Loading data...")
        pipeline.load_data(
            data_dir=data_dir,
            metabolomics_file="metabolomics.csv",
            proteomics_file="proteomics.csv",
            biochemistry_file="biochemistry.csv",
            lifestyle_file="lifestyle.csv",
            lipidomics_file="lipidomics.csv"
        )
        
        # Step 4: Run pipeline
        logger.info("Running pipeline...")
        embeddings = pipeline.run()
        
        # Step 5: Evaluate modality importance
        logger.info("Evaluating modality importance...")
        modality_importance = pipeline.evaluate_modality_importance()
        
        # Log modality importance
        logger.info("Modality importance scores:")
        for modality, score in sorted(modality_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {modality}: {score:.4f}")
        
        # Step 6: Find similar participants
        logger.info("Finding similar participants...")
        participant_id = pipeline.participant_ids[0]
        similar_participants = pipeline.find_similar_participants(participant_id, n_neighbors=5)
        
        logger.info(f"Participants similar to {participant_id}:")
        for pid, similarity in similar_participants:
            logger.info(f"  {pid}: {similarity:.4f}")
        
        # Step 7: Save results
        logger.info("Saving results...")
        pipeline.save_results(output_dir)
        
        # Step 8: Create visualizations
        logger.info("Creating visualizations...")
        
        # Participant embeddings
        visualize_participant_embeddings(
            embeddings['participant'],
            participant_ids=pipeline.participant_ids,
            method='umap',
            output_file=os.path.join(output_dir, 'participant_embeddings.png')
        )
        
        # With cluster labels
        if 'cluster_assignments' in metadata:
            # Ensure participants are in the same order
            # This is necessary because the metadata might have participants that are not in the data
            labels = []
            for pid in pipeline.participant_ids:
                idx = metadata['participant_ids'].index(pid) if pid in metadata['participant_ids'] else -1
                labels.append(metadata['cluster_assignments'][idx] if idx >= 0 else -1)
                
            visualize_participant_embeddings(
                embeddings['participant'],
                participant_ids=pipeline.participant_ids,
                method='umap',
                labels=labels,
                output_file=os.path.join(output_dir, 'participant_clusters.png')
            )
        
        # Modality importance
        plot_modality_importance(
            modality_importance,
            output_file=os.path.join(output_dir, 'modality_importance.png')
        )
        
        # Cross-modality correlations
        if 'metabolomics' in embeddings and 'proteomics' in embeddings:
            discover_cross_modality_correlations(
                pipeline,
                source_modality='metabolomics',
                target_modality='proteomics',
                correlation_threshold=0.6,
                output_file=os.path.join(output_dir, 'cross_modality_correlations.csv')
            )
        
        logger.info(f"All results saved to {output_dir}")
        logger.info("Test pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in test pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Cgraph pipeline")
    parser.add_argument("--data_dir", type=str, default="test_data", help="Directory to save test data")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results")
    parser.add_argument("--n_participants", type=int, default=100, help="Number of participants")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    
    args = parser.parse_args()
    
    success = run_test_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_participants=args.n_participants,
        n_clusters=args.n_clusters,
        embedding_dim=args.embedding_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    exit(0 if success else 1)