#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graph-based imputation module for Multi-Omics Integration Pipeline.

This module implements the GRAPEImputer class for graph-based population-level imputation,
which uses a graph neural network approach to impute missing values in omics data.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import NearestNeighbors

# Set up logging
logger = logging.getLogger(__name__)


class GRAPEImputer:
    """
    Graph-based imputation inspired by GRAPE (Graph-Based Population-Level Imputation).
    This imputer constructs a similarity graph among participants and uses a GNN to
    propagate information through the graph to impute missing values.
    """
    
    def __init__(self, n_neighbors=10, embedding_dim=128, epochs=100, lr=0.01, 
                 dropout=0.2, device=None, random_state=42):
        """
        Initialize the GRAPEImputer.
        
        Args:
            n_neighbors (int): Number of neighbors for KNN graph construction
            embedding_dim (int): Dimension of hidden embeddings
            epochs (int): Number of training epochs
            lr (float): Learning rate for optimizer
            dropout (float): Dropout probability
            device (str): Device to use (cuda or cpu)
            random_state (int): Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def create_participant_similarity_graph(self, feature_matrix):
        """
        Create a similarity graph based on available features.
        
        Args:
            feature_matrix (np.ndarray): Feature matrix with missing values (participants x features)
            
        Returns:
            torch.Tensor: Edge index tensor for the graph
        """
        logger.info("Creating participant similarity graph for imputation")
        
        # Use only complete cases for KNN
        complete_mask = ~np.isnan(feature_matrix).any(axis=1)
        complete_indices = np.where(complete_mask)[0]
        
        if len(complete_indices) < self.n_neighbors:
            logger.warning(f"Not enough complete cases for KNN. Found {len(complete_indices)}, need {self.n_neighbors}.")
            # Fall back to using cases with at least 50% non-missing values
            partial_threshold = 0.5
            partial_mask = np.isnan(feature_matrix).mean(axis=1) < partial_threshold
            partial_indices = np.where(partial_mask)[0]
            
            if len(partial_indices) < self.n_neighbors:
                raise ValueError(f"Not enough samples with {partial_threshold*100}% complete data for KNN")
            
            logger.info(f"Using {len(partial_indices)} participants with at least {partial_threshold*100}% complete data")
            
            # Fill missing values with column means for KNN
            feature_matrix_filled = feature_matrix.copy()
            col_means = np.nanmean(feature_matrix, axis=0)
            for i in range(feature_matrix.shape[1]):
                mask = np.isnan(feature_matrix[:, i])
                feature_matrix_filled[mask, i] = col_means[i]
            
            # Fit KNN on partial cases
            knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(partial_indices)))
            knn.fit(feature_matrix_filled[partial_mask])
            
            # For each participant, find neighbors
            edge_list = []
            
            # For partial cases, use KNN
            distances, indices = knn.kneighbors(feature_matrix_filled[partial_mask])
            for i, neighbors in enumerate(indices):
                src_idx = partial_indices[i]
                for j, neighbor_idx in enumerate(neighbors):
                    if src_idx != partial_indices[neighbor_idx]:  # Avoid self-loops
                        edge_list.append((src_idx, partial_indices[neighbor_idx]))
                        
            # For remaining cases, find nearest partial cases
            remaining_mask = ~partial_mask
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) > 0:
                distances, indices = knn.kneighbors(feature_matrix_filled[remaining_mask])
                for i, neighbors in enumerate(indices):
                    src_idx = remaining_indices[i]
                    for neighbor_idx in neighbors:
                        edge_list.append((src_idx, partial_indices[neighbor_idx]))
        else:
            logger.info(f"Using {len(complete_indices)} participants with complete data for KNN")
            
            # Fit KNN on complete cases
            knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(complete_indices)))
            knn.fit(feature_matrix[complete_mask])
            
            # For each participant, find neighbors among complete cases
            edge_list = []
            
            # For complete cases, use KNN
            distances, indices = knn.kneighbors(feature_matrix[complete_mask])
            for i, neighbors in enumerate(indices):
                src_idx = complete_indices[i]
                for j, neighbor_idx in enumerate(neighbors):
                    if src_idx != complete_indices[neighbor_idx]:  # Avoid self-loops
                        edge_list.append((src_idx, complete_indices[neighbor_idx]))
                        
            # For incomplete cases, find nearest complete cases
            incomplete_mask = ~complete_mask
            incomplete_indices = np.where(incomplete_mask)[0]
            
            if len(incomplete_indices) > 0:
                # Create a temporary version with NaNs replaced by means
                temp_matrix = feature_matrix.copy()
                col_means = np.nanmean(feature_matrix, axis=0)
                
                for i in range(feature_matrix.shape[1]):
                    mask = np.isnan(temp_matrix[:, i])
                    temp_matrix[mask, i] = col_means[i]
                
                # Find nearest neighbors among complete cases
                distances, indices = knn.kneighbors(temp_matrix[incomplete_mask])
                for i, neighbors in enumerate(indices):
                    src_idx = incomplete_indices[i]
                    for neighbor_idx in neighbors:
                        edge_list.append((src_idx, complete_indices[neighbor_idx]))
        
        # Add bidirectional edges to ensure information flow
        bidirectional_edge_list = []
        for src, dst in edge_list:
            bidirectional_edge_list.append((src, dst))
            bidirectional_edge_list.append((dst, src))
            
        # Remove duplicates
        bidirectional_edge_list = list(set(bidirectional_edge_list))
        
        # Convert to tensor
        edge_index = torch.tensor(bidirectional_edge_list, dtype=torch.long).t().contiguous()
        
        logger.info(f"Created similarity graph with {edge_index.shape[1]} edges")
        return edge_index
    
    def build_imputation_model(self, input_dim):
        """
        Build a GNN model for imputation.
        
        Args:
            input_dim (int): Input dimension (number of features)
            
        Returns:
            torch.nn.Module: GNN imputation model
        """
        class GNNImputer(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout):
                super(GNNImputer, self).__init__()
                self.conv1 = SAGEConv(input_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, hidden_dim)
                self.conv3 = SAGEConv(hidden_dim, output_dim)
                self.dropout = dropout
                
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.conv2(x, edge_index))
                x = self.conv3(x, edge_index)
                return x
            
        return GNNImputer(input_dim, self.embedding_dim, input_dim, self.dropout).to(self.device)
    
    def impute(self, data_df):
        """
        Impute missing values using graph neural network.
        
        Args:
            data_df (pd.DataFrame): Dataframe with missing values
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        logger.info(f"Imputing missing values for dataframe with shape {data_df.shape}")
        
        # Convert DataFrame to numpy array
        feature_matrix = data_df.values.astype(np.float32)
        
        # Create mask of missing values
        missing_mask = np.isnan(feature_matrix)
        missing_rate = missing_mask.mean()
        
        if missing_rate == 0:
            logger.info("No missing values to impute")
            return data_df
        
        logger.info(f"Missing rate: {missing_rate:.4f}")
        
        # Initialize with mean imputation
        col_means = np.nanmean(feature_matrix, axis=0)
        imputed_matrix = feature_matrix.copy()
        for i in range(feature_matrix.shape[1]):
            imputed_matrix[:, i] = np.where(
                np.isnan(feature_matrix[:, i]), 
                col_means[i], 
                feature_matrix[:, i]
            )
        
        # Create similarity graph
        try:
            edge_index = self.create_participant_similarity_graph(feature_matrix)
        except Exception as e:
            logger.error(f"Error creating similarity graph: {e}")
            logger.warning("Falling back to mean imputation")
            return pd.DataFrame(imputed_matrix, index=data_df.index, columns=data_df.columns)
        
        # Initialize and train GNN model
        model = self.build_imputation_model(feature_matrix.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Convert to PyTorch tensors
        x = torch.tensor(imputed_matrix, dtype=torch.float).to(self.device)
        edge_index = edge_index.to(self.device)
        mask = torch.tensor(~missing_mask, dtype=torch.bool).to(self.device)
        
        # Train model
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = model(x, edge_index)
            
            # Only compute loss on observed values
            loss = F.mse_loss(out[mask], x[mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
        
        # Use trained model for final imputation
        model.eval()
        with torch.no_grad():
            imputed_features = model(x, edge_index).cpu().numpy()
            
        # Replace only the missing values with imputed ones
        result_matrix = feature_matrix.copy()
        result_matrix[missing_mask] = imputed_features[missing_mask]
        
        # Create result DataFrame
        result_df = pd.DataFrame(result_matrix, index=data_df.index, columns=data_df.columns)
        
        logger.info("Imputation completed successfully")
        return result_df


class SNFImputer:
    """
    Similarity Network Fusion (SNF) based imputation for multi-omics data.
    This imputer fuses similarity networks from multiple modalities to 
    improve imputation accuracy.
    
    This class requires the 'snfpy' package to be installed.
    """
    
    def __init__(self, k=20, alpha=0.5, t=20, random_state=42):
        """
        Initialize the SNFImputer.
        
        Args:
            k (int): Number of neighbors
            alpha (float): Variance for local model
            t (int): Number of iterations
            random_state (int): Random seed for reproducibility
        """
        self.k = k
        self.alpha = alpha
        self.t = t
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Check if snfpy is installed
        try:
            import snf
            self.snf = snf
        except ImportError:
            logger.error("SNFImputer requires 'snfpy' package. Please install it with: pip install snfpy")
            raise ImportError("SNFImputer requires 'snfpy' package")
    
    def _create_similarity_matrix(self, data_matrix, missing_mask):
        """
        Create similarity matrix from data matrix with missing values.
        
        Args:
            data_matrix (np.ndarray): Data matrix (samples x features)
            missing_mask (np.ndarray): Boolean mask indicating missing values
            
        Returns:
            np.ndarray: Similarity matrix (samples x samples)
        """
        # Mean imputation for computing distance
        imputed_matrix = data_matrix.copy()
        col_means = np.nanmean(data_matrix, axis=0)
        for i in range(data_matrix.shape[1]):
            imputed_matrix[:, i] = np.where(
                missing_mask[:, i], 
                col_means[i], 
                data_matrix[:, i]
            )
        
        # Compute distance matrix
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(imputed_matrix, metric='euclidean'))
        
        # Convert to similarity matrix using SNF
        similarity_matrix = self.snf.make_affinity(dist_matrix, K=self.k, mu=self.alpha)
        
        return similarity_matrix
    
    def impute(self, modalities):
        """
        Impute missing values using SNF.
        
        Args:
            modalities (dict): Dictionary of dataframes with missing values
            
        Returns:
            dict: Dictionary of dataframes with imputed values
        """
        logger.info(f"Imputing missing values for {len(modalities)} modalities using SNF")
        
        # Ensure all modalities have the same participants
        all_participants = set.intersection(*[set(df.index) for df in modalities.values()])
        logger.info(f"Using {len(all_participants)} participants common to all modalities")
        
        # Align modalities
        aligned_modalities = {}
        for modality, df in modalities.items():
            aligned_modalities[modality] = df.loc[all_participants]
        
        # Create similarity matrices for each modality
        similarity_matrices = []
        for modality, df in aligned_modalities.items():
            data_matrix = df.values.astype(np.float32)
            missing_mask = np.isnan(data_matrix)
            
            try:
                sim_matrix = self._create_similarity_matrix(data_matrix, missing_mask)
                similarity_matrices.append(sim_matrix)
                logger.info(f"Created similarity matrix for {modality}")
            except Exception as e:
                logger.error(f"Error creating similarity matrix for {modality}: {e}")
                
        if not similarity_matrices:
            logger.error("No valid similarity matrices could be created")
            return modalities
        
        # Fuse similarity matrices using SNF
        fused_network = self.snf.snf(similarity_matrices, K=self.k, t=self.t)
        logger.info("Successfully fused similarity networks")
        
        # Perform imputation for each modality using the fused network
        imputed_modalities = {}
        
        for modality, df in aligned_modalities.items():
            data_matrix = df.values.astype(np.float32)
            missing_mask = np.isnan(data_matrix)
            
            # Skip if no missing values
            if not missing_mask.any():
                imputed_modalities[modality] = df
                continue
            
            # Impute using weighted averaging based on similarity
            imputed_matrix = data_matrix.copy()
            
            for i in range(data_matrix.shape[0]):
                for j in range(data_matrix.shape[1]):
                    if missing_mask[i, j]:
                        # Get similarities to other participants
                        similarities = fused_network[i, :]
                        
                        # Find participants with observed values for this feature
                        observed_mask = ~missing_mask[:, j]
                        
                        if observed_mask.any():
                            # Get values and similarities for observed participants
                            observed_values = data_matrix[observed_mask, j]
                            observed_similarities = similarities[observed_mask]
                            
                            # Normalize similarities
                            weights = observed_similarities / observed_similarities.sum()
                            
                            # Weighted average imputation
                            imputed_matrix[i, j] = np.sum(weights * observed_values)
                        else:
                            # If no observed values for this feature, use mean
                            imputed_matrix[i, j] = np.nanmean(data_matrix[:, j])
            
            # Create imputed dataframe
            imputed_df = pd.DataFrame(imputed_matrix, index=df.index, columns=df.columns)
            imputed_modalities[modality] = imputed_df
            
            logger.info(f"Imputed {missing_mask.sum()} missing values in {modality}")
        
        # Return imputed modalities with original participant order
        result_modalities = {}
        for modality, orig_df in modalities.items():
            if modality in imputed_modalities:
                # Align with original index
                result_df = pd.DataFrame(index=orig_df.index, columns=orig_df.columns)
                result_df.loc[imputed_modalities[modality].index] = imputed_modalities[modality]
                
                # Fill any remaining missing values with mean
                result_df = result_df.fillna(orig_df.mean())
                
                result_modalities[modality] = result_df
            else:
                # If modality wasn't imputed, return original with mean imputation
                result_modalities[modality] = orig_df.fillna(orig_df.mean())
        
        return result_modalities