#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main pipeline for multi-omics integration using heterogeneous graph neural networks.
This pipeline integrates metabolomics, proteomics, biochemistry, lifestyle, and lipidomics
data based on a graph neural network approach inspired by LLM architectures.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from tqdm import tqdm

from .src.data_processing.data_loader import load_multi_omics_data
from .src.data_processing.preprocessing import standardize_features
from .src.imputation.graph_imputer import GRAPEImputer
from .src.graph_construction.heterogeneous_graph import create_multi_omics_graph
from .src.models.hgt_model import HeterogeneousGraphTransformer, train_hgt_model
from .src.visualization.embedding_viz import visualize_participant_embeddings, analyze_attention_weights

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiOmicsIntegration:
    """
    Main class for multi-omics data integration using heterogeneous graph neural networks
    with transformer architecture inspired by Large Language Models.
    
    This pipeline integrates different omics modalities (metabolomics, proteomics, 
    biochemistry, lifestyle data, and lipidomics) to create unified embeddings that
    capture relationships across modalities.
    """
    
    def __init__(self, embedding_dim=256, n_heads=4, n_layers=2, device=None,
                 correlation_threshold=0.5, n_neighbors=15, batch_size=None, 
                 use_mini_batch=False, use_amp=False, early_stopping=False, patience=10):
        """
        Initialize the multi-omics integration pipeline.
        
        Args:
            embedding_dim (int): Dimension of embeddings 
            n_heads (int): Number of attention heads in transformer
            n_layers (int): Number of transformer layers
            device (str): Device to use (cuda or cpu). If None, uses cuda if available
            correlation_threshold (float): Threshold for creating feature-feature edges
            n_neighbors (int): Number of neighbors for graph imputation
            batch_size (int): Batch size for mini-batch training (if use_mini_batch=True)
            use_mini_batch (bool): Whether to use mini-batch training
            use_amp (bool): Whether to use automatic mixed precision
            early_stopping (bool): Whether to use early stopping
            patience (int): Patience for early stopping
        """
        # Configuration
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.correlation_threshold = correlation_threshold
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.use_mini_batch = use_mini_batch
        self.use_amp = use_amp
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize object attributes
        self.modalities = {}
        self.imputed_modalities = {}
        self.availability_df = None
        self.multi_omics_graph = None
        self.hgt_model = None
        self.embeddings = None
        self.participant_ids = None
        
    def load_data(self, data_dir=None, metabolomics_file=None, proteomics_file=None, 
                 biochemistry_file=None, lifestyle_file=None, lipidomics_file=None):
        """
        Load multi-omics data from files.
        
        Args:
            data_dir (str): Directory containing data files (optional)
            metabolomics_file (str): Path to metabolomics data
            proteomics_file (str): Path to proteomics data 
            biochemistry_file (str): Path to biochemistry data
            lifestyle_file (str): Path to lifestyle data
            lipidomics_file (str): Path to lipidomics data
            
        Returns:
            dict: Dictionary of loaded dataframes
        """
        # Construct file paths if data_dir is provided
        if data_dir:
            if metabolomics_file:
                metabolomics_file = os.path.join(data_dir, metabolomics_file)
            if proteomics_file:
                proteomics_file = os.path.join(data_dir, proteomics_file)
            if biochemistry_file:
                biochemistry_file = os.path.join(data_dir, biochemistry_file)
            if lifestyle_file:
                lifestyle_file = os.path.join(data_dir, lifestyle_file)
            if lipidomics_file:
                lipidomics_file = os.path.join(data_dir, lipidomics_file)
        
        # Load data
        self.modalities, self.availability_df = load_multi_omics_data(
            metabolomics_file=metabolomics_file,
            proteomics_file=proteomics_file,
            biochemistry_file=biochemistry_file,
            lifestyle_file=lifestyle_file,
            lipidomics_file=lipidomics_file
        )
        
        # Store participant IDs
        self.participant_ids = sorted(self.availability_df.index)
        
        logger.info(f"Loaded {len(self.modalities)} modalities for {len(self.participant_ids)} participants")
        for modality, df in self.modalities.items():
            logger.info(f"  {modality}: {df.shape[0]} samples, {df.shape[1]} features")
            
        return self.modalities
    
    def preprocess_data(self):
        """
        Standardize features across all modalities.
        
        Returns:
            dict: Dictionary of preprocessed dataframes
        """
        logger.info("Preprocessing data...")
        self.modalities = standardize_features(self.modalities)
        return self.modalities
    
    def impute_missing_data(self):
        """
        Impute missing values using graph-based imputation.
        
        Returns:
            dict: Dictionary of imputed dataframes
        """
        logger.info("Imputing missing values...")
        self.imputed_modalities = {}
        imputer = GRAPEImputer(n_neighbors=self.n_neighbors, embedding_dim=self.embedding_dim)
        
        for modality, df in self.modalities.items():
            logger.info(f"Imputing missing values for {modality}...")
            try:
                imputed_df = imputer.impute(df)
                self.imputed_modalities[modality] = imputed_df
            except Exception as e:
                logger.warning(f"Error imputing {modality}, using mean imputation instead: {e}")
                # Fallback to simple imputation
                imputed_df = df.fillna(df.mean())
                self.imputed_modalities[modality] = imputed_df
                
        return self.imputed_modalities
    
    def build_graph(self):
        """
        Build heterogeneous graph from multi-omics data.
        
        Returns:
            HeteroData: Heterogeneous graph
        """
        logger.info("Building heterogeneous graph...")
        self.multi_omics_graph = create_multi_omics_graph(
            self.imputed_modalities, 
            self.availability_df,
            correlation_threshold=self.correlation_threshold
        )
        
        # Log graph statistics
        logger.info("Graph statistics:")
        for node_type in self.multi_omics_graph.node_types:
            logger.info(f"  {node_type}: {self.multi_omics_graph[node_type].x.shape[0]} nodes")
        for edge_type in self.multi_omics_graph.edge_types:
            logger.info(f"  {edge_type}: {self.multi_omics_graph[edge_type].edge_index.shape[1]} edges")
            
        return self.multi_omics_graph
    
    def train_model(self):
        """
        Train the HGT model on the multi-omics graph.
        
        Returns:
            tuple: (model, embeddings_dict)
        """
        logger.info("Training HGT model...")
        self.hgt_model, self.embeddings = train_hgt_model(
            self.multi_omics_graph,
            hidden_channels=self.embedding_dim,
            num_heads=self.n_heads,
            num_layers=self.n_layers,
            device=self.device,
            num_epochs=200,
            lr=0.001,
            use_mini_batch=self.use_mini_batch,
            batch_size=self.batch_size,
            use_amp=self.use_amp,
            early_stopping=self.early_stopping,
            patience=self.patience
        )
        
        return self.hgt_model, self.embeddings
    
    def run(self, preprocess=True, impute=True, graph=None):
        """
        Run the full pipeline.
        
        Args:
            preprocess (bool): Whether to preprocess data
            impute (bool): Whether to impute missing data
            graph (HeteroData): Pre-constructed graph (optional)
            
        Returns:
            dict: Dictionary of embeddings
        """
        # Check if data is loaded
        if not self.modalities:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # Preprocess data if required
        if preprocess:
            self.preprocess_data()
        
        # Impute missing data if required
        if impute:
            self.impute_missing_data()
        
        # Build graph or use provided one
        if graph is not None:
            logger.info("Using provided graph...")
            self.multi_omics_graph = graph
        else:
            self.build_graph()
        
        # Train model
        self.train_model()
        
        return self.embeddings
    
    def evaluate_modality_importance(self):
        """
        Evaluate the importance of each modality by measuring its impact on embeddings.
        
        Returns:
            dict: Dictionary of modality importance scores
        """
        from .src.models.hgt_model import evaluate_modality_importance
        
        if self.hgt_model is None or self.multi_omics_graph is None:
            raise ValueError("Model not trained. Please run the pipeline first.")
        
        logger.info("Evaluating modality importance...")
        modality_importance = evaluate_modality_importance(
            self.hgt_model, 
            self.multi_omics_graph,
            device=self.device
        )
        
        # Log results
        logger.info("Modality importance scores:")
        for modality, score in sorted(modality_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {modality}: {score:.4f}")
            
        return modality_importance
    
    def find_similar_participants(self, participant_id, n_neighbors=10):
        """
        Find similar participants based on embeddings.
        
        Args:
            participant_id (str): Participant ID
            n_neighbors (int): Number of neighbors to return
            
        Returns:
            list: List of similar participant IDs with similarity scores
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available. Please run the pipeline first.")
        
        # Get participant embeddings
        participant_embeddings = self.embeddings['participant'].cpu().numpy()
        
        # Get index of the participant
        if participant_id not in self.participant_ids:
            raise ValueError(f"Participant {participant_id} not found in data.")
        
        participant_idx = self.participant_ids.index(participant_id)
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([participant_embeddings[participant_idx]], participant_embeddings)[0]
        
        # Get top n_neighbors (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_neighbors+1]
        similar_participants = [
            (self.participant_ids[idx], similarities[idx])
            for idx in similar_indices
        ]
        
        return similar_participants
    
    def compute_subgroup_enrichment(self, subgroups):
        """
        Compute enriched features in each subgroup.
        
        Args:
            subgroups (dict): Dictionary mapping participant IDs to subgroup labels
            
        Returns:
            dict: Dictionary of enriched features per subgroup
        """
        from scipy import stats
        
        if not self.imputed_modalities:
            raise ValueError("Imputed data not available. Please run the pipeline first.")
        
        # Convert subgroups to label array
        subgroup_labels = np.array([subgroups.get(pid, -1) for pid in self.participant_ids])
        unique_subgroups = sorted(set(subgroup_labels))
        
        # Remove -1 (unmapped participants)
        if -1 in unique_subgroups:
            unique_subgroups.remove(-1)
        
        enrichment = {}
        
        # For each modality
        for modality, df in self.imputed_modalities.items():
            modality_enrichment = {}
            
            # Align dataframe with participant_ids
            aligned_df = df.reindex(self.participant_ids)
            
            # For each subgroup
            for subgroup in unique_subgroups:
                subgroup_mask = (subgroup_labels == subgroup)
                other_mask = (subgroup_labels != subgroup) & (subgroup_labels != -1)
                
                # Skip if not enough samples
                if sum(subgroup_mask) < 5 or sum(other_mask) < 5:
                    continue
                
                # Compute t-tests for each feature
                pvals = []
                effect_sizes = []
                feature_names = []
                
                for feature in aligned_df.columns:
                    subgroup_values = aligned_df.loc[subgroup_mask, feature].dropna()
                    other_values = aligned_df.loc[other_mask, feature].dropna()
                    
                    # Skip if not enough values
                    if len(subgroup_values) < 5 or len(other_values) < 5:
                        continue
                    
                    # Compute t-test
                    t_stat, p_val = stats.ttest_ind(subgroup_values, other_values, equal_var=False)
                    
                    # Compute effect size (Cohen's d)
                    mean1, mean2 = subgroup_values.mean(), other_values.mean()
                    std1, std2 = subgroup_values.std(), other_values.std()
                    pooled_std = np.sqrt(((len(subgroup_values) - 1) * std1**2 + 
                                         (len(other_values) - 1) * std2**2) / 
                                        (len(subgroup_values) + len(other_values) - 2))
                    
                    effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    pvals.append(p_val)
                    effect_sizes.append(effect_size)
                    feature_names.append(feature)
                
                # Adjust p-values for multiple testing
                if pvals:
                    from statsmodels.stats.multitest import multipletests
                    adj_pvals = multipletests(pvals, method='fdr_bh')[1]
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'feature': feature_names,
                        'effect_size': effect_sizes,
                        'p_value': pvals,
                        'adj_p_value': adj_pvals
                    })
                    
                    # Filter for significance
                    sig_results = results_df[results_df['adj_p_value'] < 0.05].sort_values(
                        by='effect_size', ascending=False
                    )
                    
                    modality_enrichment[subgroup] = sig_results
            
            enrichment[modality] = modality_enrichment
        
        return enrichment
    
    def save_results(self, output_dir):
        """
        Save model, embeddings, and graph.
        
        Args:
            output_dir (str): Output directory
        """
        if self.hgt_model is None or self.embeddings is None or self.multi_omics_graph is None:
            raise ValueError("No results to save. Please run the pipeline first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state dict
        torch.save(self.hgt_model.state_dict(), os.path.join(output_dir, 'hgt_model.pt'))
        
        # Save embeddings
        for node_type, emb in self.embeddings.items():
            torch.save(emb, os.path.join(output_dir, f'{node_type}_embeddings.pt'))
        
        # Save graph structure
        torch.save(self.multi_omics_graph, os.path.join(output_dir, 'multi_omics_graph.pt'))
        
        # Save participant IDs and availability
        with open(os.path.join(output_dir, 'participant_ids.pkl'), 'wb') as f:
            pickle.dump(self.participant_ids, f)
            
        if self.availability_df is not None:
            self.availability_df.to_csv(os.path.join(output_dir, 'modality_availability.csv'))
        
        logger.info(f"Results saved to {output_dir}")
        
    def load_model(self, model_path, graph_path=None):
        """
        Load a saved model and optionally a graph.
        
        Args:
            model_path (str): Path to saved model
            graph_path (str): Path to saved graph (optional)
            
        Returns:
            torch.nn.Module: Loaded model
        """
        # Load graph if provided
        if graph_path:
            self.multi_omics_graph = torch.load(graph_path, map_location=self.device)
            logger.info(f"Loaded graph from {graph_path}")
        
        if self.multi_omics_graph is None:
            raise ValueError("No graph available. Please load a graph or run the pipeline first.")
        
        # Initialize model
        self.hgt_model = HeterogeneousGraphTransformer(
            self.multi_omics_graph,
            hidden_channels=self.embedding_dim,
            num_heads=self.n_heads,
            num_layers=self.n_layers
        ).to(self.device)
        
        # Load model state dict
        self.hgt_model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Loaded model from {model_path}")
        
        return self.hgt_model


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Omics Integration Pipeline")
    parser.add_argument("--data_dir", type=str, help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--correlation_threshold", type=float, default=0.5, help="Correlation threshold")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for imputation")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--use_mini_batch", action="store_true", help="Use mini-batch training")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MultiOmicsIntegration(
        embedding_dim=args.embedding_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        correlation_threshold=args.correlation_threshold,
        n_neighbors=args.n_neighbors,
        batch_size=args.batch_size,
        use_mini_batch=args.use_mini_batch,
        use_amp=args.use_amp,
        early_stopping=args.early_stopping,
        patience=args.patience,
        device=args.device
    )
    
    # Load data
    if args.data_dir:
        pipeline.load_data(
            data_dir=args.data_dir,
            metabolomics_file="metabolomics.csv",
            proteomics_file="proteomics.csv",
            biochemistry_file="biochemistry.csv",
            lifestyle_file="lifestyle.csv",
            lipidomics_file="lipidomics.csv"
        )
        
        # Run pipeline
        pipeline.run()
        
        # Evaluate modality importance
        pipeline.evaluate_modality_importance()
        
        # Save results
        pipeline.save_results(args.output_dir)
    else:
        parser.print_help()