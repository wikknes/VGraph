#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading module for Multi-Omics Integration Pipeline.
"""

import os
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


def load_multi_omics_data(
    metabolomics_file=None,
    proteomics_file=None,
    biochemistry_file=None,
    lifestyle_file=None,
    lipidomics_file=None
):
    """
    Load multi-omics data from CSV files and create an availability matrix.
    
    Args:
        metabolomics_file (str): Path to metabolomics data CSV
        proteomics_file (str): Path to proteomics data CSV
        biochemistry_file (str): Path to biochemistry data CSV
        lifestyle_file (str): Path to lifestyle data CSV
        lipidomics_file (str): Path to lipidomics data CSV
        
    Returns:
        tuple: (modalities_dict, availability_df)
            - modalities_dict: Dictionary of dataframes for each modality
            - availability_df: DataFrame indicating which participants have which modalities
    """
    # Dictionary to track modality availability per participant
    modality_presence = defaultdict(dict)
    
    # Load and inventory all CSV files
    modalities = {}
    modality_names = []
    modality_files = {}
    
    if metabolomics_file:
        modality_names.append('metabolomics')
        modality_files['metabolomics'] = metabolomics_file
    if proteomics_file:
        modality_names.append('proteomics')
        modality_files['proteomics'] = proteomics_file
    if biochemistry_file:
        modality_names.append('biochemistry')
        modality_files['biochemistry'] = biochemistry_file
    if lifestyle_file:
        modality_names.append('lifestyle')
        modality_files['lifestyle'] = lifestyle_file
    if lipidomics_file:
        modality_names.append('lipidomics')
        modality_files['lipidomics'] = lipidomics_file
        
    if not modality_names:
        raise ValueError("No data files provided. Please provide at least one modality file.")
        
    all_lab_ids = set()
    
    for modality in modality_names:
        try:
            file_path = modality_files[modality]
            logger.info(f"Loading {modality} data from {file_path}")
            
            df = pd.read_csv(file_path)
            # Ensure lab_ID column exists and is correctly named
            id_col = [col for col in df.columns if col.lower() in ['lab_id', 'labid', 'id', 'subject_id', 'subject', 'participant_id', 'participant']]
            
            if not id_col:
                raise ValueError(f"No ID column found in {modality} data. Expected 'lab_ID' or similar.")
                
            id_col = id_col[0]
            if id_col != 'lab_ID':
                logger.info(f"Renaming '{id_col}' to 'lab_ID' in {modality} data")
                df.rename(columns={id_col: 'lab_ID'}, inplace=True)
                
            # Track which participants have which modalities
            for lab_id in df['lab_ID']:
                modality_presence[lab_id][modality] = True
                all_lab_ids.add(lab_id)
                
            # Set lab_ID as index
            df.set_index('lab_ID', inplace=True)
            
            modalities[modality] = df
            logger.info(f"Loaded {modality}: {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            logger.error(f"Error loading {modality}: {e}")
    
    # Create summary of modality availability
    availability_df = pd.DataFrame(index=sorted(all_lab_ids), columns=modality_names)
    for lab_id in all_lab_ids:
        for modality in modality_names:
            availability_df.loc[lab_id, modality] = modality in modality_presence[lab_id]
    
    logger.info(f"Total unique participants: {len(all_lab_ids)}")
    logger.info(f"Participants with all modalities: {availability_df.all(axis=1).sum()}")
    
    # Save availability matrix for debugging
    try:
        availability_df.to_csv("modality_availability.csv")
        logger.info("Saved modality availability matrix to 'modality_availability.csv'")
    except Exception as e:
        logger.warning(f"Could not save modality availability matrix: {e}")
    
    return modalities, availability_df


def preprocess_omics_data(
    df,
    id_col=None,
    missing_threshold=0.3,
    min_variance=0.01,
    log_transform=True,
    scaling="z-score"
):
    """
    Preprocess omics data by handling missing values, transforming, and scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        id_col (str): Name of the ID column to convert to lab_ID
        missing_threshold (float): Remove features with missingness above this threshold
        min_variance (float): Remove features with variance below this threshold
        log_transform (bool): Whether to apply log transformation to positive values
        scaling (str): Scaling method ("z-score", "min-max", or None)
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # If id_col is provided, rename to lab_ID
    if id_col is not None and id_col in df.columns and id_col != "lab_ID":
        df = df.rename(columns={id_col: "lab_ID"})
    
    # Make a copy of the original dataframe
    processed_df = df.copy()
    
    # Separate ID column if not already set as index
    if "lab_ID" in processed_df.columns:
        processed_df.set_index("lab_ID", inplace=True)
    
    # Calculate missingness per feature
    missingness = processed_df.isnull().mean()
    
    # Remove features with too many missing values
    high_missing_features = missingness[missingness > missing_threshold].index
    if len(high_missing_features) > 0:
        logger.info(f"Removing {len(high_missing_features)} features with >{missing_threshold*100:.1f}% missing values")
        processed_df = processed_df.drop(columns=high_missing_features)
    
    # Calculate variance per feature
    variance = processed_df.var()
    
    # Remove low-variance features
    low_var_features = variance[variance < min_variance].index
    if len(low_var_features) > 0:
        logger.info(f"Removing {len(low_var_features)} features with variance <{min_variance}")
        processed_df = processed_df.drop(columns=low_var_features)
    
    # Apply log transformation if requested
    if log_transform:
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            # Apply log transform to positive values (with offset for zeros)
            mask = processed_df[col] > 0
            if mask.any():
                min_positive = processed_df.loc[mask, col].min() / 10
                processed_df.loc[mask, col] = np.log1p(processed_df.loc[mask, col] - min_positive)
    
    # Apply scaling if requested
    if scaling == "z-score":
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            processed_df[col] = (processed_df[col] - processed_df[col].mean()) / (processed_df[col].std() + 1e-8)
    elif scaling == "min-max":
        numeric_cols = processed_df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            min_val = processed_df[col].min()
            max_val = processed_df[col].max()
            if max_val > min_val:
                processed_df[col] = (processed_df[col] - min_val) / (max_val - min_val)
    
    # Reset index to get lab_ID as a column
    if "lab_ID" not in processed_df.columns:
        processed_df = processed_df.reset_index()
    
    logger.info(f"Preprocessed dataframe: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
    return processed_df