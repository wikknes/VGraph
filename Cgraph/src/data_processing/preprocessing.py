#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing module for Multi-Omics Integration Pipeline.
"""

import logging
import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


def standardize_features(modalities):
    """
    Standardize features across all modalities.
    
    Args:
        modalities (dict): Dictionary of dataframes for each modality
        
    Returns:
        dict: Dictionary of standardized dataframes
    """
    standardized_modalities = {}
    
    for modality, df in modalities.items():
        logger.info(f"Standardizing features for {modality}")
        
        # Make a copy of the dataframe
        std_df = df.copy()
        
        # Handle different modalities appropriately
        if modality == 'lifestyle':
            # One-hot encode categorical variables
            cat_cols = std_df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                logger.info(f"One-hot encoding {len(cat_cols)} categorical variables in lifestyle data")
                std_df = pd.get_dummies(std_df, columns=cat_cols, dummy_na=False)
        else:
            # For omics data: log transform positive values, then standardize
            numeric_cols = std_df.select_dtypes(include=np.number).columns
            logger.info(f"Standardizing {len(numeric_cols)} numeric columns in {modality} data")
            
            for col in numeric_cols:
                # Apply log transform to positive values (with offset for zeros)
                mask = std_df[col] > 0
                if mask.any():
                    min_positive = std_df.loc[mask, col].min() / 10
                    std_df.loc[mask, col] = np.log1p(std_df.loc[mask, col] - min_positive)
                
                # Z-score normalization
                mean_val = std_df[col].mean()
                std_val = std_df[col].std()
                if not np.isnan(mean_val) and not np.isnan(std_val) and std_val > 0:
                    std_df[col] = (std_df[col] - mean_val) / (std_val + 1e-8)
        
        standardized_modalities[modality] = std_df
    
    return standardized_modalities


def handle_outliers(modalities, method='winsorize', threshold=3.0):
    """
    Handle outliers in the data.
    
    Args:
        modalities (dict): Dictionary of dataframes for each modality
        method (str): Method to handle outliers ('winsorize', 'cap', or 'remove')
        threshold (float): Threshold for outlier detection (number of standard deviations)
        
    Returns:
        dict: Dictionary of dataframes with outliers handled
    """
    from scipy import stats
    
    cleaned_modalities = {}
    
    for modality, df in modalities.items():
        logger.info(f"Handling outliers in {modality} data")
        
        # Make a copy of the dataframe
        clean_df = df.copy()
        
        # Only apply to numeric columns
        numeric_cols = clean_df.select_dtypes(include=np.number).columns
        
        if method == 'winsorize':
            # Winsorize outliers (replace with threshold values)
            for col in numeric_cols:
                clean_df[col] = stats.mstats.winsorize(clean_df[col], limits=[0.05, 0.05])
                
        elif method == 'cap':
            # Cap outliers at threshold standard deviations
            for col in numeric_cols:
                mean_val = clean_df[col].mean()
                std_val = clean_df[col].std()
                
                if not np.isnan(mean_val) and not np.isnan(std_val) and std_val > 0:
                    upper_limit = mean_val + threshold * std_val
                    lower_limit = mean_val - threshold * std_val
                    
                    # Cap values
                    clean_df[col] = np.where(clean_df[col] > upper_limit, upper_limit, clean_df[col])
                    clean_df[col] = np.where(clean_df[col] < lower_limit, lower_limit, clean_df[col])
                    
        elif method == 'remove':
            # Identify outliers
            for col in numeric_cols:
                mean_val = clean_df[col].mean()
                std_val = clean_df[col].std()
                
                if not np.isnan(mean_val) and not np.isnan(std_val) and std_val > 0:
                    upper_limit = mean_val + threshold * std_val
                    lower_limit = mean_val - threshold * std_val
                    
                    # Set outliers to NaN
                    clean_df.loc[(clean_df[col] > upper_limit) | (clean_df[col] < lower_limit), col] = np.nan
        
        cleaned_modalities[modality] = clean_df
    
    return cleaned_modalities


def filter_low_variance_features(modalities, min_variance=0.01):
    """
    Filter out low-variance features from all modalities.
    
    Args:
        modalities (dict): Dictionary of dataframes for each modality
        min_variance (float): Minimum variance threshold
        
    Returns:
        dict: Dictionary of filtered dataframes
    """
    filtered_modalities = {}
    
    for modality, df in modalities.items():
        logger.info(f"Filtering low-variance features from {modality} data")
        
        # Calculate variance for each feature
        variances = df.var()
        low_var_features = variances[variances < min_variance].index
        
        # Filter out low-variance features
        if len(low_var_features) > 0:
            logger.info(f"Removing {len(low_var_features)} low-variance features from {modality}")
            filtered_df = df.drop(columns=low_var_features)
        else:
            filtered_df = df
            
        filtered_modalities[modality] = filtered_df
    
    return filtered_modalities


def batch_correct(modalities, batch_col='batch_id'):
    """
    Apply batch correction to omics data.
    
    Args:
        modalities (dict): Dictionary of dataframes for each modality
        batch_col (str): Column name containing batch information
        
    Returns:
        dict: Dictionary of batch-corrected dataframes
    """
    try:
        import combat
        has_combat = True
    except ImportError:
        logger.warning("combat package not found. Using simple batch correction method.")
        has_combat = False
    
    corrected_modalities = {}
    
    for modality, df in modalities.items():
        # Skip lifestyle data or data without batch information
        if modality == 'lifestyle' or batch_col not in df.columns:
            corrected_modalities[modality] = df
            continue
            
        logger.info(f"Applying batch correction to {modality} data")
        
        if has_combat:
            # Use ComBat for batch correction
            batch_ids = df[batch_col].values
            data_matrix = df.drop(columns=[batch_col]).values.T  # ComBat expects genes x samples
            
            corrected_matrix = combat.combat(data_matrix, batch_ids)
            corrected_df = pd.DataFrame(
                corrected_matrix.T,  # Convert back to samples x features
                index=df.index,
                columns=df.columns.drop(batch_col)
            )
        else:
            # Simple batch correction: center each batch
            corrected_df = df.copy()
            feature_cols = [col for col in df.columns if col != batch_col]
            
            for batch in df[batch_col].unique():
                batch_mask = df[batch_col] == batch
                
                # Center each feature within the batch
                for col in feature_cols:
                    batch_mean = df.loc[batch_mask, col].mean()
                    corrected_df.loc[batch_mask, col] = df.loc[batch_mask, col] - batch_mean
        
        corrected_modalities[modality] = corrected_df
    
    return corrected_modalities


def merge_redundant_features(modalities, correlation_threshold=0.95, min_samples=10):
    """
    Merge highly correlated features to reduce redundancy.
    
    Args:
        modalities (dict): Dictionary of dataframes for each modality
        correlation_threshold (float): Correlation threshold for merging
        min_samples (int): Minimum number of samples required to compute correlation
        
    Returns:
        dict: Dictionary of dataframes with merged features
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    merged_modalities = {}
    
    for modality, df in modalities.items():
        logger.info(f"Checking for redundant features in {modality} data")
        
        # Skip if too few samples or features
        if df.shape[0] < min_samples or df.shape[1] < 2:
            merged_modalities[modality] = df
            continue
        
        # Compute correlation matrix
        corr_matrix = df.corr().abs().fillna(0)
        
        # Convert to distance matrix
        distance_matrix = 1 - corr_matrix
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(squareform(distance_matrix), method='average')
        
        # Form clusters at the correlation threshold
        clusters = hierarchy.fcluster(linkage, 1 - correlation_threshold, criterion='distance')
        
        # Group features by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(df.columns[i])
        
        # Merge features within each cluster
        merged_df = pd.DataFrame(index=df.index)
        merged_features_count = 0
        
        for cluster_id, features in cluster_groups.items():
            if len(features) == 1:
                # Single feature, no merging needed
                merged_df[features[0]] = df[features[0]]
            else:
                # Multiple features, take mean
                merged_feature_name = f"{modality}_cluster_{cluster_id}"
                merged_df[merged_feature_name] = df[features].mean(axis=1)
                merged_features_count += len(features) - 1
        
        if merged_features_count > 0:
            logger.info(f"Merged {merged_features_count} redundant features in {modality}")
            
        merged_modalities[modality] = merged_df
    
    return merged_modalities