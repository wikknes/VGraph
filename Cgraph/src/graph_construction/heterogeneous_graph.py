#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Heterogeneous graph construction module for Multi-Omics Integration Pipeline.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import networkx as nx

# Set up logging
logger = logging.getLogger(__name__)


def create_multi_omics_graph(modalities, availability_df, correlation_threshold=0.5):
    """
    Create a heterogeneous graph with participants and features as nodes.
    
    Args:
        modalities (dict): Dictionary of dataframes for each modality
        availability_df (pd.DataFrame): DataFrame indicating which participants have which modalities
        correlation_threshold (float): Threshold for creating feature-feature edges
        
    Returns:
        HeteroData: Heterogeneous graph
    """
    logger.info("Creating heterogeneous multi-omics graph")
    
    graph = HeteroData()
    
    # Add participant nodes (all participants)
    all_participants = sorted(availability_df.index)
    n_participants = len(all_participants)
    participant_id_map = {pid: i for i, pid in enumerate(all_participants)}
    
    # Initialize participant features (demographic/clinical if available, otherwise zeros)
    if 'lifestyle' in modalities:
        # Use lifestyle data as participant features
        lifestyle_df = modalities['lifestyle']
        participant_features = np.zeros((n_participants, lifestyle_df.shape[1]))
        
        for i, pid in enumerate(all_participants):
            if pid in lifestyle_df.index:
                participant_features[i] = lifestyle_df.loc[pid].values
    else:
        # Use simple one-hot encoding if no lifestyle data
        participant_features = np.eye(n_participants)
    
    # Add participant nodes to graph
    graph['participant'].x = torch.tensor(participant_features, dtype=torch.float)
    graph['participant'].lab_ids = all_participants
    
    logger.info(f"Added {n_participants} participant nodes with {participant_features.shape[1]} features")
    
    # For each modality, add feature nodes and edges
    feature_offset = 0
    
    for modality_name in modalities:
        df = modalities[modality_name]
        feature_names = df.columns.tolist()
        n_features = len(feature_names)
        
        # Add feature nodes for this modality
        feature_id_map = {fname: i + feature_offset for i, fname in enumerate(feature_names)}
        graph[modality_name].x = torch.eye(n_features, dtype=torch.float)  # One-hot encoding
        graph[modality_name].feature_names = feature_names
        
        logger.info(f"Added {n_features} feature nodes for {modality_name}")
        
        # Add participant-to-feature edges (with values as edge attributes)
        edge_indices = []
        edge_attrs = []
        
        # Get the number of participants and features
        n_participants = graph['participant'].x.size(0)
        
        for pid in all_participants:
            if pid in df.index:  # Participant has this modality
                p_idx = participant_id_map[pid]
                # Make sure participant index is valid
                if p_idx >= n_participants:
                    continue
                    
                for feature_name in feature_names:
                    f_idx = feature_id_map[feature_name]
                    # Make sure feature index is valid
                    if f_idx >= n_features:
                        continue
                        
                    value = df.loc[pid, feature_name]
                    if not np.isnan(value):
                        edge_indices.append((p_idx, f_idx))
                        edge_attrs.append(value)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
            
            # Add edges to graph
            graph['participant', f'has_{modality_name}', modality_name].edge_index = edge_index
            graph['participant', f'has_{modality_name}', modality_name].edge_attr = edge_attr
            
            logger.info(f"Added {len(edge_indices)} edges between participants and {modality_name} features")
        
        # Add feature-to-feature edges (based on correlation)
        if len(df) > 10:  # Need enough samples for meaningful correlation
            # Compute correlation matrix
            corr_matrix = df.corr().abs().fillna(0)
            feature_edges = []
            
            for i, f1 in enumerate(feature_names):
                for j, f2 in enumerate(feature_names):
                    if i < j and corr_matrix.loc[f1, f2] > correlation_threshold:
                        f1_idx = feature_id_map[f1]
                        f2_idx = feature_id_map[f2]
                        feature_edges.append((f1_idx, f2_idx))
                        feature_edges.append((f2_idx, f1_idx))  # Add both directions
            
            if feature_edges:
                feature_edge_index = torch.tensor(feature_edges, dtype=torch.long).t().contiguous()
                graph[modality_name, 'correlated_with', modality_name].edge_index = feature_edge_index
                
                logger.info(f"Added {len(feature_edges)} correlation edges within {modality_name} features")
        
        feature_offset += n_features
    
    # Add cross-modality feature-feature edges (if requested)
    add_cross_modality_edges(graph, modalities, correlation_threshold)
    
    return graph


def add_cross_modality_edges(graph, modalities, correlation_threshold=0.5, min_samples=10):
    """
    Add edges between features from different modalities based on their associations.
    
    Args:
        graph (HeteroData): Heterogeneous graph
        modalities (dict): Dictionary of dataframes for each modality
        correlation_threshold (float): Threshold for creating feature-feature edges
        min_samples (int): Minimum number of overlapping samples required
        
    Returns:
        HeteroData: Updated heterogeneous graph
    """
    # Get all pairs of modalities
    modality_names = list(modalities.keys())
    if len(modality_names) < 2:
        return graph
    
    # For each pair of modalities
    for i, mod1 in enumerate(modality_names):
        for j, mod2 in enumerate(modality_names):
            if i >= j:  # Skip same modality and reversed pairs
                continue
            
            df1 = modalities[mod1]
            df2 = modalities[mod2]
            
            # Find participants with both modalities
            common_participants = list(set(df1.index).intersection(set(df2.index)))
            
            if len(common_participants) < min_samples:
                logger.info(f"Not enough common participants between {mod1} and {mod2}, skipping cross-modality edges")
                continue
            
            # Align dataframes
            aligned_df1 = df1.loc[common_participants]
            aligned_df2 = df2.loc[common_participants]
            
            # Compute correlations between features
            try:
                cross_corr = pd.DataFrame(
                    np.corrcoef(aligned_df1.T, aligned_df2.T)[:len(aligned_df1.columns), len(aligned_df1.columns):],
                    index=aligned_df1.columns,
                    columns=aligned_df2.columns
                ).abs()
                
                # Create edges for correlations above threshold
                feature_edges = []
                edge_attrs = []
                
                # Get feature indices
                mod1_feature_names = graph[mod1].feature_names
                mod2_feature_names = graph[mod2].feature_names
                mod1_indices = {name: i for i, name in enumerate(mod1_feature_names)}
                mod2_indices = {name: i for i, name in enumerate(mod2_feature_names)}
                
                # Make sure we track the total number of nodes correctly
                mod1_node_count = graph[mod1].x.size(0)
                mod2_node_count = graph[mod2].x.size(0)
                
                # Find correlated feature pairs
                for f1 in cross_corr.index:
                    for f2 in cross_corr.columns:
                        corr_val = cross_corr.loc[f1, f2]
                        if corr_val > correlation_threshold:
                            if f1 in mod1_indices and f2 in mod2_indices:
                                f1_idx = mod1_indices[f1]
                                f2_idx = mod2_indices[f2]
                                
                                # Make sure indices are valid
                                if f1_idx < mod1_node_count and f2_idx < mod2_node_count:
                                    feature_edges.append((f1_idx, f2_idx))
                                    edge_attrs.append(corr_val)
                
                if feature_edges:
                    # Add edges to graph
                    edge_index = torch.tensor(feature_edges, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
                    
                    edge_type = (mod1, f'associated_with', mod2)
                    graph[edge_type].edge_index = edge_index
                    graph[edge_type].edge_attr = edge_attr
                    
                    # Also add reverse edges
                    reverse_edges = [(dst, src) for src, dst in feature_edges]
                    reverse_edge_index = torch.tensor(reverse_edges, dtype=torch.long).t().contiguous()
                    
                    reverse_edge_type = (mod2, f'associated_with', mod1)
                    graph[reverse_edge_type].edge_index = reverse_edge_index
                    graph[reverse_edge_type].edge_attr = edge_attr.clone()
                    
                    logger.info(f"Added {len(feature_edges)} cross-modality edges between {mod1} and {mod2}")
                
            except Exception as e:
                logger.error(f"Error computing cross-modality edges between {mod1} and {mod2}: {e}")
    
    return graph


def add_prior_knowledge_edges(graph, interactions_df, source_col, target_col, modality_col=None, 
                             weight_col=None, edge_type='interacts_with'):
    """
    Add edges based on prior knowledge (e.g., from pathway databases).
    
    Args:
        graph (HeteroData): Heterogeneous graph
        interactions_df (pd.DataFrame): DataFrame with interaction information
        source_col (str): Column name for source features
        target_col (str): Column name for target features
        modality_col (str): Column name for modality information
        weight_col (str): Column name for edge weights
        edge_type (str): Edge type name
        
    Returns:
        HeteroData: Updated heterogeneous graph
    """
    logger.info("Adding prior knowledge edges to graph")
    
    # Get all feature names and their modalities from graph
    feature_to_modality = {}
    feature_to_index = {}
    
    for modality in graph.node_types:
        if modality == 'participant':
            continue
            
        if hasattr(graph[modality], 'feature_names'):
            for i, fname in enumerate(graph[modality].feature_names):
                feature_to_modality[fname] = modality
                feature_to_index[(modality, fname)] = i
    
    # Process interactions
    edges_added = 0
    
    if modality_col in interactions_df.columns:
        # Group by source and target modalities
        for (src_modality, tgt_modality), group in interactions_df.groupby([f"{source_col}_{modality_col}", f"{target_col}_{modality_col}"]):
            if src_modality not in graph.node_types or tgt_modality not in graph.node_types:
                continue
                
            edge_lists = {}
            edge_attrs = {}
            
            for _, row in group.iterrows():
                src_feature = row[source_col]
                tgt_feature = row[target_col]
                
                # Skip if features not in graph
                if (src_modality, src_feature) not in feature_to_index or (tgt_modality, tgt_feature) not in feature_to_index:
                    continue
                    
                src_idx = feature_to_index[(src_modality, src_feature)]
                tgt_idx = feature_to_index[(tgt_modality, tgt_feature)]
                
                edge_type_key = (src_modality, edge_type, tgt_modality)
                
                if edge_type_key not in edge_lists:
                    edge_lists[edge_type_key] = []
                    edge_attrs[edge_type_key] = []
                
                edge_lists[edge_type_key].append((src_idx, tgt_idx))
                
                # Add edge weight if available
                if weight_col is not None:
                    edge_attrs[edge_type_key].append(row[weight_col])
                else:
                    edge_attrs[edge_type_key].append(1.0)
            
            # Add edges to graph
            for edge_type_key, edges in edge_lists.items():
                if not edges:
                    continue
                    
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs[edge_type_key], dtype=torch.float).view(-1, 1)
                
                graph[edge_type_key].edge_index = edge_index
                graph[edge_type_key].edge_attr = edge_attr
                
                edges_added += len(edges)
                
                # Also add reverse edges if applicable
                if edge_type_key[0] != edge_type_key[2]:  # Different node types
                    reverse_edge_type = (edge_type_key[2], edge_type, edge_type_key[0])
                    reverse_edges = [(dst, src) for src, dst in edges]
                    
                    reverse_edge_index = torch.tensor(reverse_edges, dtype=torch.long).t().contiguous()
                    
                    graph[reverse_edge_type].edge_index = reverse_edge_index
                    graph[reverse_edge_type].edge_attr = edge_attr.clone()
                    
                    edges_added += len(reverse_edges)
    else:
        # Without modality column, try to match features to modalities
        new_edges = {}
        new_edge_attrs = {}
        
        for _, row in interactions_df.iterrows():
            src_feature = row[source_col]
            tgt_feature = row[target_col]
            
            # Skip if features not in graph
            if src_feature not in feature_to_modality or tgt_feature not in feature_to_modality:
                continue
                
            src_modality = feature_to_modality[src_feature]
            tgt_modality = feature_to_modality[tgt_feature]
            
            src_idx = feature_to_index[(src_modality, src_feature)]
            tgt_idx = feature_to_index[(tgt_modality, tgt_feature)]
            
            edge_type_key = (src_modality, edge_type, tgt_modality)
            
            if edge_type_key not in new_edges:
                new_edges[edge_type_key] = []
                new_edge_attrs[edge_type_key] = []
            
            new_edges[edge_type_key].append((src_idx, tgt_idx))
            
            # Add edge weight if available
            if weight_col is not None:
                new_edge_attrs[edge_type_key].append(row[weight_col])
            else:
                new_edge_attrs[edge_type_key].append(1.0)
        
        # Add edges to graph
        for edge_type_key, edges in new_edges.items():
            if not edges:
                continue
                
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(new_edge_attrs[edge_type_key], dtype=torch.float).view(-1, 1)
            
            graph[edge_type_key].edge_index = edge_index
            graph[edge_type_key].edge_attr = edge_attr
            
            edges_added += len(edges)
            
            # Also add reverse edges if applicable
            if edge_type_key[0] != edge_type_key[2]:  # Different node types
                reverse_edge_type = (edge_type_key[2], edge_type, edge_type_key[0])
                reverse_edges = [(dst, src) for src, dst in edges]
                
                reverse_edge_index = torch.tensor(reverse_edges, dtype=torch.long).t().contiguous()
                
                graph[reverse_edge_type].edge_index = reverse_edge_index
                graph[reverse_edge_type].edge_attr = edge_attr.clone()
                
                edges_added += len(reverse_edges)
    
    logger.info(f"Added {edges_added} edges from prior knowledge")
    return graph


def visualize_graph(graph, output_file=None, max_nodes=100):
    """
    Visualize the heterogeneous graph using NetworkX.
    
    Args:
        graph (HeteroData): Heterogeneous graph
        output_file (str): Path to save the visualization
        max_nodes (int): Maximum number of nodes to visualize
        
    Returns:
        nx.Graph: NetworkX graph
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        node_offset = 0
        node_colors = {}
        node_labels = {}
        
        for node_type in graph.node_types:
            num_nodes = graph[node_type].x.shape[0]
            
            # Limit number of nodes if too many
            if num_nodes > max_nodes:
                logger.warning(f"Limiting {node_type} nodes to {max_nodes} for visualization")
                sample_indices = np.random.choice(num_nodes, max_nodes, replace=False)
            else:
                sample_indices = range(num_nodes)
            
            # Add nodes with type as attribute
            for i in sample_indices:
                node_id = f"{node_type}_{i}"
                G.add_node(node_id, node_type=node_type)
                
                # Set color by node type
                node_colors[node_id] = {
                    'participant': 'skyblue',
                    'metabolomics': 'salmon',
                    'proteomics': 'lightgreen',
                    'biochemistry': 'orange',
                    'lifestyle': 'purple',
                    'lipidomics': 'pink'
                }.get(node_type, 'gray')
                
                # Set label
                if node_type == 'participant' and hasattr(graph[node_type], 'lab_ids'):
                    node_labels[node_id] = str(graph[node_type].lab_ids[i])
                elif hasattr(graph[node_type], 'feature_names'):
                    feat_name = graph[node_type].feature_names[i]
                    node_labels[node_id] = feat_name[:10] + '...' if len(feat_name) > 10 else feat_name
                else:
                    node_labels[node_id] = node_id
        
        # Add edges
        for edge_type in graph.edge_types:
            src_type, edge_name, dst_type = edge_type
            
            edge_index = graph[edge_type].edge_index
            
            for j in range(edge_index.shape[1]):
                src_idx = edge_index[0, j].item()
                dst_idx = edge_index[1, j].item()
                
                src_id = f"{src_type}_{src_idx}"
                dst_id = f"{dst_type}_{dst_idx}"
                
                # Only add edges between nodes that are in the graph
                if src_id in G and dst_id in G:
                    G.add_edge(src_id, dst_id, edge_type=edge_name)
        
        # Plot graph
        plt.figure(figsize=(12, 10))
        
        # Use spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        for node_type in set(nx.get_node_attributes(G, 'node_type').values()):
            node_list = [node for node, data in G.nodes(data=True) if data['node_type'] == node_type]
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=node_list,
                node_color=[node_colors[node] for node in node_list],
                node_size=100,
                label=node_type
            )
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        plt.title("Heterogeneous Multi-Omics Graph")
        plt.legend()
        plt.axis('off')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
        
        return G
        
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        return None