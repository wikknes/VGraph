#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pseudo data generator for Cgraph.

This module provides functions to generate realistic pseudo data for multi-omics
integration. It creates synthetic datasets that mimic real-world omics data with
realistic distributions, missing patterns, and correlations.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)


def generate_pseudo_omics_data(
    output_dir,
    n_participants=200,
    n_clusters=3,
    modalities=("metabolomics", "proteomics", "biochemistry", "lifestyle", "lipidomics"),
    feature_counts=None,
    missing_rates=None,
    cluster_sep=2.0,
    add_correlated_features=True,
    add_noise_level=0.2,
    include_binary_features=True,
    include_categorical_features=True,
    random_state=42
):
    """
    Generate high-quality pseudo data for multi-omics integration.
    
    This function creates realistic synthetic datasets with known ground truth
    clustering patterns, cross-modality correlations, and realistic noise and
    missing data patterns.
    
    Args:
        output_dir (str): Directory to save the generated data
        n_participants (int): Number of participants to generate
        n_clusters (int): Number of participant clusters to generate
        modalities (tuple): List of modalities to generate
        feature_counts (dict): Number of features per modality
        missing_rates (dict): Rate of missing data per modality
        cluster_sep (float): Separation between clusters (higher = more distinct)
        add_correlated_features (bool): Whether to add correlated features
        add_noise_level (float): Level of noise to add (0-1)
        include_binary_features (bool): Whether to include binary features
        include_categorical_features (bool): Whether to include categorical features
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with metadata about the generated data
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_state)
    
    # Default feature counts if not provided
    if feature_counts is None:
        feature_counts = {
            'metabolomics': 100,
            'proteomics': 200,
            'biochemistry': 50,
            'lifestyle': 30,
            'lipidomics': 150
        }
    
    # Default missing rates if not provided
    if missing_rates is None:
        missing_rates = {
            'metabolomics': 0.10,
            'proteomics': 0.15,
            'biochemistry': 0.05,
            'lifestyle': 0.02,
            'lipidomics': 0.20
        }
    
    # Verify that all requested modalities have feature counts
    for modality in modalities:
        if modality not in feature_counts:
            logger.warning(f"No feature count specified for {modality}, using default of 50")
            feature_counts[modality] = 50
        if modality not in missing_rates:
            logger.warning(f"No missing rate specified for {modality}, using default of 0.1")
            missing_rates[modality] = 0.1
    
    # Generate participant IDs and cluster assignments
    participant_ids = [f'SUBJ{i:04d}' for i in range(n_participants)]
    
    # Create cluster assignments for participants
    cluster_assignments, cluster_centers = make_blobs(
        n_samples=n_participants,
        n_features=2,  # 2D embedding for visualization
        centers=n_clusters,
        cluster_std=1.0,
        center_box=(-10, 10),
        shuffle=True,
        random_state=random_state,
        return_centers=True
    )
    
    # Normalize to 0-1 range for easy visualization
    scaler = StandardScaler()
    cluster_assignments = scaler.fit_transform(cluster_assignments)
    
    # Convert to cluster labels
    y_true = np.argmin(
        np.sum((cluster_assignments[:, np.newaxis, :] - 
               cluster_centers[np.newaxis, :, :]) ** 2, axis=2),
        axis=1
    )
    
    # Track which participants have which modalities
    availability = np.random.rand(n_participants, len(modalities))
    # Ensure at least 80% data availability
    availability = availability > np.percentile(availability, 20, axis=0)
    
    # Create a metadata dictionary to track ground truth
    metadata = {
        'participant_ids': participant_ids,
        'cluster_assignments': y_true,
        'modalities': {},
        'cross_modality_correlations': []
    }
    
    # Generate data for each modality
    for i, modality in enumerate(modalities):
        # Get participants with this modality
        available_mask = availability[:, i]
        available_participants = [pid for j, pid in enumerate(participant_ids) if available_mask[j]]
        avail_idx = [j for j, _ in enumerate(participant_ids) if available_mask[j]]
        
        n_features = feature_counts.get(modality, 50)
        missing_rate = missing_rates.get(modality, 0.1)
        
        # Generate different data types based on modality
        if modality == 'lifestyle':
            # Create lifestyle data with mix of continuous, binary and categorical
            df = _generate_lifestyle_data(
                avail_idx, y_true, n_features, include_binary_features,
                include_categorical_features, random_state
            )
        else:
            # Generate omics data using classification to ensure biological relevance
            X, feature_importances = _generate_omics_data(
                avail_idx, y_true, n_features, n_clusters, 
                add_correlated_features, add_noise_level, random_state
            )
            
            # Track feature importance for ground truth
            metadata['modalities'][modality] = {
                'feature_importances': feature_importances
            }
            
            # Create modality-specific transformations
            if modality == 'metabolomics':
                # Metabolomics data typically right-skewed and positive
                X = np.exp(X + np.random.normal(0, 0.1, X.shape))
            elif modality == 'proteomics':
                # Proteomics often has high dynamic range
                X = np.exp(X * 2)
            elif modality == 'lipidomics':
                # Lipidomics often has high zeros and positive skew
                X[X < -1] = 0
                X = np.exp(X)
            
            # Add missing values
            mask = np.random.rand(*X.shape) < missing_rate
            X[mask] = np.nan
            
            # Create column names
            columns = [f'{modality}_{j}' for j in range(n_features)]
            
            # Create dataframe
            df = pd.DataFrame(X, columns=columns, index=available_participants)
        
        # Save to CSV
        file_path = os.path.join(output_dir, f'{modality}.csv')
        df.reset_index().rename(columns={'index': 'lab_ID'}).to_csv(file_path, index=False)
        logger.info(f"Generated {modality} data: {df.shape[0]} participants, {df.shape[1]} features")
    
    # Generate cross-modality correlations for biological realism
    if len(modalities) >= 2 and add_correlated_features:
        _add_cross_modality_correlations(output_dir, modalities, metadata)
    
    # Save metadata for ground truth
    metadata_file = os.path.join(output_dir, 'ground_truth.csv')
    pd.DataFrame({
        'lab_ID': participant_ids,
        'cluster': y_true
    }).to_csv(metadata_file, index=False)
    
    logger.info(f"Pseudo data generation complete. Files saved to {output_dir}")
    return metadata


def _generate_lifestyle_data(avail_idx, y_true, n_features, include_binary, include_categorical, random_state):
    """Generate lifestyle data with mixed data types."""
    n_participants = len(avail_idx)
    y_avail = np.array([y_true[idx] for idx in avail_idx])
    
    # Determine feature distribution
    n_continuous = n_features // 2
    n_binary = n_features // 4 if include_binary else 0
    n_categorical = n_features - n_continuous - n_binary if include_categorical else 0
    
    data = {}
    
    # Generate continuous variables
    for j in range(n_continuous):
        # Some variables correlated with clusters
        if j < n_continuous // 3:
            # Strongly cluster-associated
            values = y_avail + np.random.normal(0, 0.5, n_participants)
            scaling = np.random.uniform(0.5, 2.0)
            values = values * scaling + np.random.uniform(-5, 5)
        else:
            # Random continuous variables
            values = np.random.normal(0, 1, n_participants)
            
        data[f'lifestyle_continuous_{j}'] = values
    
    # Generate binary variables
    for j in range(n_binary):
        if j < n_binary // 3:
            # Cluster-associated binary
            probs = np.array([0.2, 0.5, 0.8])[y_avail]
            values = np.random.binomial(1, probs)
        else:
            # Random binary
            values = np.random.binomial(1, 0.5, n_participants)
            
        data[f'lifestyle_binary_{j}'] = values
    
    # Generate categorical variables
    for j in range(n_categorical):
        if j < n_categorical // 3:
            # Cluster-associated categorical (3 levels)
            # Cluster 0 - mostly 'Low', Cluster 1 - mostly 'Medium', Cluster 2 - mostly 'High'
            probs = np.zeros((n_participants, 3))
            for c in range(3):
                cluster_mask = y_avail == c
                probs[cluster_mask, c] = 0.7
                probs[cluster_mask, (c+1)%3] = 0.2
                probs[cluster_mask, (c+2)%3] = 0.1
                
            values = np.array(['Low', 'Medium', 'High'])[
                np.array([np.random.choice(3, p=probs[i]) for i in range(n_participants)])
            ]
        else:
            # Random categorical
            values = np.random.choice(['Low', 'Medium', 'High'], size=n_participants)
            
        data[f'lifestyle_categorical_{j}'] = values
    
    return pd.DataFrame(data)


def _generate_omics_data(avail_idx, y_true, n_features, n_clusters, add_correlated, noise_level, random_state):
    """Generate omics data with realistic patterns."""
    n_participants = len(avail_idx)
    y_avail = np.array([y_true[idx] for idx in avail_idx])
    
    # Determine informative features (25% of features are strongly cluster-associated)
    n_informative = max(int(n_features * 0.25), 5)
    n_redundant = max(int(n_features * 0.15), 3) if add_correlated else 0
    n_repeated = max(int(n_features * 0.05), 2) if add_correlated else 0
    
    # Create data using classification to ensure cluster patterns
    X, _ = make_classification(
        n_samples=n_participants,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_clusters,
        n_clusters_per_class=1,
        weights=None,
        flip_y=noise_level,
        class_sep=2.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=random_state
    )
    
    # Create sample-specific noise scale to simulate biological variation
    noise_scales = np.random.uniform(0.5, 1.5, n_participants)
    
    # Add realistic noise
    for i in range(n_participants):
        noise = np.random.normal(0, noise_scales[i] * noise_level, n_features)
        X[i, :] += noise
    
    # Calculate feature importance based on how well each separates clusters
    feature_importances = []
    for j in range(n_features):
        f_vals = []
        for c in range(n_clusters):
            if np.sum(y_avail == c) > 1:  # Need at least 2 samples
                f_vals.append(np.mean(X[y_avail == c, j]))
        
        # Higher variance between clusters = more important feature
        if len(f_vals) > 1:
            importance = np.var(f_vals)
        else:
            importance = 0
        feature_importances.append(importance)
    
    # Normalize importances
    if max(feature_importances) > 0:
        feature_importances = np.array(feature_importances) / max(feature_importances)
    else:
        feature_importances = np.zeros(n_features)
    
    return X, feature_importances


def _add_cross_modality_correlations(output_dir, modalities, metadata):
    """Add correlations between features from different modalities."""
    # Load all modality data
    data = {}
    for modality in modalities:
        file_path = os.path.join(output_dir, f'{modality}.csv')
        df = pd.read_csv(file_path)
        df.set_index('lab_ID', inplace=True)
        data[modality] = df
    
    # Create cross-modality correlations for biological realism
    correlations = []
    
    # Choose random pairs of modalities
    for i, mod1 in enumerate(modalities):
        if mod1 == 'lifestyle':
            continue  # Skip lifestyle as source
            
        for j, mod2 in enumerate(modalities):
            if i == j or mod2 == 'lifestyle':
                continue  # Skip same modality or lifestyle as target
                
            df1 = data[mod1]
            df2 = data[mod2]
            
            # Find common participants
            common_participants = set(df1.index).intersection(set(df2.index))
            if len(common_participants) < 10:
                continue
                
            # Select random source features (20% of features or max 10)
            n_source = min(10, max(1, int(df1.shape[1] * 0.2)))
            source_features = np.random.choice(df1.columns, n_source, replace=False)
            
            # Select random target features (20% of features or max 10)
            n_target = min(10, max(1, int(df2.shape[1] * 0.2)))
            target_features = np.random.choice(df2.columns, n_target, replace=False)
            
            # Create correlations
            for src_feat in source_features:
                src_vals = df1.loc[common_participants, src_feat].copy()
                
                # Skip if too many missing values
                if src_vals.isna().sum() > len(src_vals) * 0.3:
                    continue
                    
                # Fill missing with mean for correlation calculation
                src_vals = src_vals.fillna(src_vals.mean())
                
                for tgt_feat in target_features:
                    tgt_vals = df2.loc[common_participants, tgt_feat].copy()
                    
                    # Skip if too many missing values
                    if tgt_vals.isna().sum() > len(tgt_vals) * 0.3:
                        continue
                        
                    # Fill missing with mean for correlation calculation
                    tgt_vals = tgt_vals.fillna(tgt_vals.mean())
                    
                    # Create correlation with random strength and noise
                    corr_strength = np.random.uniform(0.6, 0.9)
                    noise_level = np.random.uniform(0.2, 0.4)
                    
                    # Standardize source
                    src_std = (src_vals - src_vals.mean()) / (src_vals.std() + 1e-8)
                    
                    # Create correlated target
                    tgt_new = src_std * corr_strength + np.random.normal(0, noise_level, len(src_std))
                    
                    # Transform back to original scale
                    tgt_new = tgt_new * tgt_vals.std() + tgt_vals.mean()
                    
                    # Update target values while preserving missing values
                    missing_mask = df2.loc[common_participants, tgt_feat].isna()
                    df2.loc[common_participants, tgt_feat] = tgt_new
                    df2.loc[missing_mask.index[missing_mask], tgt_feat] = np.nan
                    
                    # Calculate actual correlation after noise addition
                    actual_corr = np.corrcoef(
                        src_vals.values, 
                        tgt_new[~np.isnan(tgt_new)]
                    )[0,1]
                    
                    # Record correlation metadata
                    correlations.append({
                        'source_modality': mod1,
                        'source_feature': src_feat,
                        'target_modality': mod2,
                        'target_feature': tgt_feat,
                        'correlation': actual_corr
                    })
    
    # Update files with correlated data
    for modality in modalities:
        if modality in data:
            file_path = os.path.join(output_dir, f'{modality}.csv')
            data[modality].reset_index().to_csv(file_path, index=False)
    
    # Save correlation information
    if correlations:
        corr_df = pd.DataFrame(correlations)
        corr_file = os.path.join(output_dir, 'cross_modality_correlations.csv')
        corr_df.to_csv(corr_file, index=False)
        
        metadata['cross_modality_correlations'] = correlations
        logger.info(f"Added {len(correlations)} cross-modality correlations")


def generate_high_accuracy_data(
    output_dir,
    n_participants=500,
    class_proportions=(0.6, 0.3, 0.1),
    n_clusters=3,
    cluster_separation=3.0,
    high_signal_ratio=0.4,
    random_state=42
):
    """
    Generate high accuracy pseudo data with strong signals and clear cluster separation.
    
    This function creates synthetic data with high signal-to-noise ratio,
    making it easier for models to achieve high accuracy.
    
    Args:
        output_dir (str): Directory to save the generated data
        n_participants (int): Number of participants to generate
        class_proportions (tuple): Proportions of participants in each class
        n_clusters (int): Number of participant clusters to generate
        cluster_separation (float): Separation between clusters (higher = more distinct)
        high_signal_ratio (float): Ratio of features with strong signals (0-1)
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with metadata about the generated data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_state)
    
    # Feature counts with more signal features
    feature_counts = {
        'metabolomics': 80,
        'proteomics': 120,
        'biochemistry': 40,
        'lifestyle': 20,
        'lipidomics': 100
    }
    
    # Lower missing rates for higher quality data
    missing_rates = {
        'metabolomics': 0.05,
        'proteomics': 0.08,
        'biochemistry': 0.03,
        'lifestyle': 0.01,
        'lipidomics': 0.10
    }
    
    # Increase feature importance
    metadata = generate_pseudo_omics_data(
        output_dir=output_dir,
        n_participants=n_participants,
        n_clusters=n_clusters,
        feature_counts=feature_counts,
        missing_rates=missing_rates,
        cluster_sep=cluster_separation,
        add_correlated_features=True,
        add_noise_level=0.15,  # Lower noise
        include_binary_features=True,
        include_categorical_features=True,
        random_state=random_state
    )
    
    logger.info(f"High accuracy data generation complete. Files saved to {output_dir}")
    return metadata


def main():
    """Command-line entry point for generating pseudo data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pseudo data for multi-omics integration")
    parser.add_argument("--output_dir", type=str, default="pseudo_data", help="Output directory")
    parser.add_argument("--n_participants", type=int, default=200, help="Number of participants")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--high_accuracy", action="store_true", help="Generate high accuracy data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.high_accuracy:
        logger.info("Generating high accuracy pseudo data...")
        generate_high_accuracy_data(
            output_dir=args.output_dir,
            n_participants=args.n_participants,
            n_clusters=args.n_clusters,
            random_state=args.seed
        )
    else:
        logger.info("Generating standard pseudo data...")
        generate_pseudo_omics_data(
            output_dir=args.output_dir,
            n_participants=args.n_participants,
            n_clusters=args.n_clusters,
            random_state=args.seed
        )
    

if __name__ == "__main__":
    main()