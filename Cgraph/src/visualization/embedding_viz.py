#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for Multi-Omics Integration Pipeline.
"""

import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set up logging
logger = logging.getLogger(__name__)


def visualize_participant_embeddings(embeddings, participant_ids=None, method='umap', 
                                    color_by=None, color_data=None, output_file=None, 
                                    normalize=True, scale_range=(-10, 10)):
    """
    Visualize participant embeddings using dimensionality reduction.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Participant embeddings
        participant_ids (list): List of participant IDs
        method (str): Dimensionality reduction method ('umap', 'tsne', or 'pca')
        color_by (str): Column in color_data to color points by
        color_data (pd.DataFrame): DataFrame with data for coloring
        output_file (str): Path to save the visualization
        normalize (bool): Whether to normalize embeddings before dimensionality reduction
        scale_range (tuple): Range to scale normalized embeddings to
        
    Returns:
        tuple: (plt.Figure, reduced_embeddings)
    """
    logger.info(f"Visualizing participant embeddings using {method}")
    
    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Normalize embeddings if requested
    if normalize:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # First use StandardScaler to center and scale to unit variance
        embeddings_np = StandardScaler().fit_transform(embeddings_np)
        
        # Then scale to desired range if specified
        if scale_range:
            min_val, max_val = scale_range
            embeddings_np = MinMaxScaler(feature_range=scale_range).fit_transform(embeddings_np)
            
        logger.info(f"Normalized embeddings to range [{np.min(embeddings_np):.2f}, {np.max(embeddings_np):.2f}]")
    
    # Apply dimensionality reduction
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_np)
        except ImportError:
            logger.warning("UMAP not installed. Falling back to t-SNE.")
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_np)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_np)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_np)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1]
    })
    
    # Add participant IDs if provided
    if participant_ids is not None:
        plot_df['participant_id'] = participant_ids
    
    # Add color data if provided
    if color_by is not None and color_data is not None:
        # Make sure participant_ids are available
        if participant_ids is None:
            logger.warning("Cannot color by data without participant_ids")
        else:
            # Create a mapping from participant IDs to color values
            color_map = {}
            for i, pid in enumerate(participant_ids):
                if pid in color_data.index and color_by in color_data.columns:
                    color_map[pid] = color_data.loc[pid, color_by]
            
            # Add color values to plot dataframe
            plot_df['color'] = plot_df['participant_id'].map(color_map)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    if 'color' in plot_df.columns:
        # Check if color is categorical or continuous
        if pd.api.types.is_numeric_dtype(plot_df['color']):
            scatter = plt.scatter(
                plot_df['x'], 
                plot_df['y'], 
                c=plot_df['color'], 
                cmap='viridis', 
                s=30, 
                alpha=0.8
            )
            plt.colorbar(scatter, label=color_by)
        else:
            # Categorical coloring
            categories = plot_df['color'].dropna().unique()
            cmap = plt.get_cmap('tab10', len(categories))
            
            for i, category in enumerate(categories):
                mask = plot_df['color'] == category
                plt.scatter(
                    plot_df.loc[mask, 'x'], 
                    plot_df.loc[mask, 'y'], 
                    color=cmap(i), 
                    label=category,
                    s=30, 
                    alpha=0.8
                )
            
            plt.legend(title=color_by)
    else:
        # Simple scatter plot
        plt.scatter(plot_df['x'], plot_df['y'], s=30, alpha=0.8)
    
    plt.title(f'Participant Embeddings ({method.upper()})')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    
    return plt.gcf(), embedding_2d


def visualize_feature_embeddings(embeddings, feature_names=None, modality=None, 
                                method='umap', output_file=None):
    """
    Visualize feature embeddings using dimensionality reduction.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Feature embeddings
        feature_names (list): List of feature names
        modality (str): Modality name for title
        method (str): Dimensionality reduction method ('umap', 'tsne', or 'pca')
        output_file (str): Path to save the visualization
        
    Returns:
        tuple: (plt.Figure, reduced_embeddings)
    """
    logger.info(f"Visualizing feature embeddings for {modality}")
    
    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Apply dimensionality reduction
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_np)
        except ImportError:
            logger.warning("UMAP not installed. Falling back to t-SNE.")
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_np)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_np)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_np)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=30, alpha=0.8)
    
    # Add feature names as annotations if provided
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            # Only annotate some points to avoid overcrowding
            if i % max(1, len(feature_names) // 100) == 0:
                plt.annotate(
                    name,
                    (embedding_2d[i, 0], embedding_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )
    
    modality_str = f" ({modality})" if modality else ""
    plt.title(f'Feature Embeddings{modality_str} ({method.upper()})')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    
    return plt.gcf(), embedding_2d


def plot_modality_importance(modality_importance, output_file=None):
    """
    Plot modality importance scores.
    
    Args:
        modality_importance (dict): Dictionary of modality importance scores
        output_file (str): Path to save the visualization
        
    Returns:
        plt.Figure: Figure object
    """
    # Sort modalities by importance
    sorted_modalities = sorted(modality_importance.items(), key=lambda x: x[1], reverse=True)
    modalities = [m[0] for m in sorted_modalities]
    scores = [m[1] for m in sorted_modalities]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modalities, scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02 * max(scores),
            f'{height:.3f}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    plt.title('Modality Importance')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    
    return plt.gcf()


def analyze_attention_weights(model, graph):
    """
    Extract and analyze attention weights from HGT model.
    
    Args:
        model (torch.nn.Module): HGT model
        graph (HeteroData): Heterogeneous graph
        
    Returns:
        dict: Dictionary of attention weights
    """
    attention_dict = {}
    
    # Extract attention weights from each layer
    for i, conv in enumerate(model.convs):
        for edge_type in graph.edge_types:
            # Get attention weights for this edge type
            src, rel, dst = edge_type
            if hasattr(conv, 'attention'):
                att = conv.attention[(src, rel, dst)]
                if att is not None:
                    attention_dict[f'layer{i}_{src}_{rel}_{dst}'] = att.detach().cpu()
    
    # Analyze and log important connections
    logger.info("Analyzing attention weights...")
    
    for att_name, att_weight in attention_dict.items():
        if att_weight.numel() > 0:  # Check if tensor is not empty
            mean_att = att_weight.mean(dim=0)  # Average over all instances
            logger.info(f"Attention analysis for {att_name}:")
            logger.info(f"Mean attention: {mean_att}")
            logger.info(f"Max attention: {mean_att.max().item()}")
    
    return attention_dict


def visualize_subgroups(embeddings, subgroups, enrichment=None, output_file=None):
    """
    Visualize participant subgroups based on embeddings.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Participant embeddings
        subgroups (dict): Dictionary mapping participant IDs to subgroup labels
        enrichment (dict): Dictionary of enriched features per subgroup
        output_file (str): Path to save the visualization
        
    Returns:
        tuple: (plt.Figure, reduced_embeddings)
    """
    logger.info("Visualizing participant subgroups")
    
    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Apply UMAP for dimensionality reduction
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_np)
    except ImportError:
        logger.warning("UMAP not installed. Falling back to t-SNE.")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_np)
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'participant_id': list(subgroups.keys()),
        'subgroup': list(subgroups.values())
    })
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Get unique subgroups and assign colors
    unique_subgroups = sorted(set(subgroups.values()))
    cmap = plt.get_cmap('tab10', len(unique_subgroups))
    
    # Plot each subgroup
    for i, subgroup in enumerate(unique_subgroups):
        mask = plot_df['subgroup'] == subgroup
        plt.scatter(
            plot_df.loc[mask, 'x'], 
            plot_df.loc[mask, 'y'], 
            color=cmap(i), 
            label=f'Subgroup {subgroup}',
            s=50, 
            alpha=0.8
        )
    
    plt.title('Participant Subgroups (UMAP)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title="Subgroups")
    
    # Add annotation for enriched features if provided
    if enrichment is not None:
        enrichment_text = "Enriched Features by Subgroup:\n\n"
        
        for modality, modality_enrichment in enrichment.items():
            enrichment_text += f"{modality.upper()}:\n"
            
            for subgroup, features_df in modality_enrichment.items():
                if len(features_df) > 0:
                    top_features = features_df.head(3)
                    enrichment_text += f"  Subgroup {subgroup}: "
                    enrichment_text += ", ".join(top_features['feature']) + "\n"
        
        # Add text box with enrichment info
        plt.figtext(
            1.02, 0.5, enrichment_text,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            fontsize=9,
            verticalalignment='center'
        )
        
        # Adjust figure size to accommodate the text
        plt.subplots_adjust(right=0.75)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    
    return plt.gcf(), embedding_2d


def visualize_participant_similarity(embeddings, participant_ids, similarity_threshold=0.7, 
                                    output_file=None, max_participants=200):
    """
    Visualize participant similarity as a network.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Participant embeddings
        participant_ids (list): List of participant IDs
        similarity_threshold (float): Threshold for similarity to create an edge
        output_file (str): Path to save the visualization
        max_participants (int): Maximum number of participants to include
        
    Returns:
        plt.Figure: Figure object
    """
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    
    logger.info("Visualizing participant similarity network")
    
    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Sample participants if too many
    if len(participant_ids) > max_participants:
        logger.info(f"Sampling {max_participants} participants for visualization")
        indices = np.random.choice(len(participant_ids), max_participants, replace=False)
        embeddings_np = embeddings_np[indices]
        participant_ids = [participant_ids[i] for i in indices]
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings_np)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, pid in enumerate(participant_ids):
        G.add_node(pid)
    
    # Add edges for similar participants
    for i in range(len(participant_ids)):
        for j in range(i+1, len(participant_ids)):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                G.add_edge(participant_ids[i], participant_ids[j], weight=similarity)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    logger.info(f"Created similarity network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get connected components
    components = list(nx.connected_components(G))
    logger.info(f"Network has {len(components)} connected components")
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Calculate node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes, colored by component
    for i, component in enumerate(components):
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(component),
            node_color=f'C{i%10}',
            node_size=50,
            alpha=0.8,
            label=f'Group {i+1} ({len(component)} participants)'
        )
    
    # Draw edges with transparency based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=1.0,
        alpha=0.5,
        edge_color=weights,
        edge_cmap=plt.cm.Blues
    )
    
    # Add some labels for large components
    for component in components:
        if len(component) > len(participant_ids) / 20:  # Only label large components
            # Pick a central node in the component
            central_node = min(component, key=lambda n: sum(nx.shortest_path_length(G, n, target) 
                                                          for target in component))
            nx.draw_networkx_labels(
                G, pos,
                labels={central_node: central_node},
                font_size=8,
                font_color='black'
            )
    
    plt.title('Participant Similarity Network')
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1)
    plt.axis('off')
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    
    return plt.gcf()


def discover_cross_modality_correlations(pipeline, source_modality, target_modality,
                                        correlation_threshold=0.5, output_file=None,
                                        ignore_missing_features=True):
    """
    Discover correlations between features from different modalities.

    Args:
        pipeline (MultiOmicsIntegration): Pipeline object
        source_modality (str): Source modality name
        target_modality (str): Target modality name
        correlation_threshold (float): Threshold for correlation
        output_file (str): Path to save the results
        ignore_missing_features (bool): Whether to continue if feature names are missing

    Returns:
        list: List of correlation dictionaries
    """
    logger.info(f"Discovering correlations between {source_modality} and {target_modality}")

    # Get embeddings
    if pipeline.embeddings is None:
        logger.error("Embeddings not available. Please run the pipeline first.")
        # If output file is requested, create an empty one
        if output_file:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
            logger.warning(f"No embeddings available. Saved empty correlations file to {output_file}")
        return []

    # Check if modalities exist in embeddings
    if source_modality not in pipeline.embeddings:
        logger.error(f"Source modality {source_modality} not found in embeddings")
        # If output file is requested, create an empty one
        if output_file:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
        return []

    if target_modality not in pipeline.embeddings:
        logger.error(f"Target modality {target_modality} not found in embeddings")
        # If output file is requested, create an empty one
        if output_file:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
        return []

    source_embeddings = pipeline.embeddings[source_modality]
    target_embeddings = pipeline.embeddings[target_modality]

    # Convert to numpy if needed
    if isinstance(source_embeddings, torch.Tensor):
        source_embeddings = source_embeddings.detach().cpu().numpy()
    if isinstance(target_embeddings, torch.Tensor):
        target_embeddings = target_embeddings.detach().cpu().numpy()

    # Check for NaN values
    if np.isnan(source_embeddings).any() or np.isnan(target_embeddings).any():
        logger.error("NaN values found in embeddings. Cannot compute correlations.")
        # If output file is requested, create an empty one
        if output_file:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
            logger.warning(f"NaN values in embeddings. Saved empty correlations file to {output_file}")
        return []

    # Check if embeddings are empty
    if source_embeddings.size == 0 or target_embeddings.size == 0:
        logger.error("Empty embeddings found. Cannot compute correlations.")
        # If output file is requested, create an empty one
        if output_file:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
            logger.warning(f"Empty embeddings. Saved empty correlations file to {output_file}")
        return []

    # Get feature names or generate placeholder names if not available
    try:
        if hasattr(pipeline.multi_omics_graph[source_modality], 'feature_names'):
            source_features = pipeline.multi_omics_graph[source_modality].feature_names
        else:
            if ignore_missing_features:
                logger.warning(f"Feature names not found for {source_modality}, using placeholders")
                source_features = [f"{source_modality}_feature_{i}" for i in range(source_embeddings.shape[0])]
            else:
                raise ValueError(f"Feature names not found for {source_modality}")
    except (AttributeError, KeyError) as e:
        if ignore_missing_features:
            logger.warning(f"Feature names not found for {source_modality}, using placeholders: {e}")
            source_features = [f"{source_modality}_feature_{i}" for i in range(source_embeddings.shape[0])]
        else:
            logger.error(f"Feature names not found for {source_modality}: {e}")
            # If output file is requested, create an empty one
            if output_file:
                empty_df = pd.DataFrame(columns=[
                    'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
                ])
                empty_df.to_csv(output_file, index=False)
            return []

    try:
        if hasattr(pipeline.multi_omics_graph[target_modality], 'feature_names'):
            target_features = pipeline.multi_omics_graph[target_modality].feature_names
        else:
            if ignore_missing_features:
                logger.warning(f"Feature names not found for {target_modality}, using placeholders")
                target_features = [f"{target_modality}_feature_{i}" for i in range(target_embeddings.shape[0])]
            else:
                raise ValueError(f"Feature names not found for {target_modality}")
    except (AttributeError, KeyError) as e:
        if ignore_missing_features:
            logger.warning(f"Feature names not found for {target_modality}, using placeholders: {e}")
            target_features = [f"{target_modality}_feature_{i}" for i in range(target_embeddings.shape[0])]
        else:
            logger.error(f"Feature names not found for {target_modality}: {e}")
            # If output file is requested, create an empty one
            if output_file:
                empty_df = pd.DataFrame(columns=[
                    'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
                ])
                empty_df.to_csv(output_file, index=False)
            return []

    # Ensure we have feature names with correct length
    if len(source_features) != source_embeddings.shape[0]:
        logger.warning(f"Number of source features ({len(source_features)}) does not match embeddings shape ({source_embeddings.shape[0]}), adjusting")
        source_features = [f"{source_modality}_feature_{i}" for i in range(source_embeddings.shape[0])]

    if len(target_features) != target_embeddings.shape[0]:
        logger.warning(f"Number of target features ({len(target_features)}) does not match embeddings shape ({target_embeddings.shape[0]}), adjusting")
        target_features = [f"{target_modality}_feature_{i}" for i in range(target_embeddings.shape[0])]

    # Try to normalize embeddings for better similarity comparison
    try:
        from sklearn.preprocessing import normalize
        source_embeddings = normalize(source_embeddings)
        target_embeddings = normalize(target_embeddings)
    except Exception as e:
        logger.warning(f"Error normalizing embeddings: {e}. Using unnormalized embeddings.")

    # Compute similarity between feature embeddings
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
    except Exception as e:
        logger.error(f"Error computing similarity matrix: {e}")
        # If output file is requested, create an empty one
        if output_file:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
            logger.warning(f"Error computing similarities. Saved empty correlations file to {output_file}")
        return []

    # Find significant correlations
    correlations = []

    # Lower the threshold if no correlations found at the original threshold
    current_threshold = correlation_threshold
    max_attempts = 3  # Maximum number of threshold reductions
    attempt = 0

    while attempt < max_attempts:
        for i in range(len(source_features)):
            for j in range(len(target_features)):
                try:
                    similarity = similarity_matrix[i, j]
                    if not np.isnan(similarity) and similarity > current_threshold:
                        correlations.append({
                            'source_modality': source_modality,
                            'target_modality': target_modality,
                            'source_feature': source_features[i],
                            'target_feature': target_features[j],
                            'score': float(similarity)  # Ensure it's a native Python float
                        })
                except Exception as e:
                    logger.warning(f"Error processing correlation for features {i},{j}: {e}")
                    continue

        # If we found correlations or used the minimum threshold, break
        if correlations or current_threshold <= 0.2:  # Don't go below 0.2
            break

        # Reduce threshold and try again
        current_threshold -= 0.1
        attempt += 1
        logger.warning(f"No correlations found at threshold {current_threshold + 0.1}, trying {current_threshold}")

    # If we still don't have correlations, force at least one
    if not correlations and similarity_matrix.size > 0:
        # Find the highest similarity pair
        flat_idx = np.nanargmax(similarity_matrix)
        if not np.isnan(flat_idx):
            i, j = np.unravel_index(flat_idx, similarity_matrix.shape)
            best_similarity = similarity_matrix[i, j]

            # Only add if it's valid
            if not np.isnan(best_similarity):
                logger.warning(f"Forcing one correlation with highest similarity score: {best_similarity}")
                correlations.append({
                    'source_modality': source_modality,
                    'target_modality': target_modality,
                    'source_feature': source_features[i],
                    'target_feature': target_features[j],
                    'score': float(best_similarity)
                })

    # Sort by similarity score
    correlations.sort(key=lambda x: x['score'], reverse=True)

    # Save to file if requested
    if output_file:
        try:
            df = pd.DataFrame(correlations)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(correlations)} correlations to {output_file}")
        except Exception as e:
            logger.error(f"Error saving correlations to file: {e}")

            # Save empty dataframe in case of error
            try:
                empty_df = pd.DataFrame(columns=[
                    'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
                ])
                empty_df.to_csv(output_file, index=False)
            except Exception as e2:
                logger.error(f"Error saving empty dataframe: {e2}")

        # If no correlations found, write an empty dataframe with correct columns
        if not correlations:
            empty_df = pd.DataFrame(columns=[
                'source_modality', 'target_modality', 'source_feature', 'target_feature', 'score'
            ])
            empty_df.to_csv(output_file, index=False)
            logger.warning(f"No correlations found above threshold {correlation_threshold}. Consider lowering the threshold.")

    return correlations


def discover_subgroups(embeddings, n_clusters=5, method='leiden', resolution=1.0):
    """
    Discover subgroups of participants based on embeddings.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Participant embeddings
        n_clusters (int): Number of clusters (for k-means and hierarchical)
        method (str): Clustering method ('leiden', 'kmeans', or 'hierarchical')
        resolution (float): Resolution parameter for Leiden clustering
        
    Returns:
        dict: Dictionary mapping participant indices to subgroup labels
    """
    logger.info(f"Discovering participant subgroups using {method}")
    
    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Apply clustering
    if method == 'leiden':
        try:
            import networkx as nx
            from scipy.spatial.distance import pdist, squareform
            
            # Compute similarity graph
            similarity = 1 - squareform(pdist(embeddings_np, metric='cosine'))
            similarity[similarity < 0.5] = 0  # Sparsify
            
            # Create graph
            G = nx.from_numpy_array(similarity)
            
            # Apply Leiden clustering
            try:
                import leidenalg as la
                import igraph as ig
                
                # Convert to igraph
                g = ig.Graph.from_networkx(G)
                
                # Run Leiden algorithm
                partition = la.find_partition(
                    g, 
                    la.ModularityVertexPartition, 
                    resolution_parameter=resolution
                )
                
                subgroups = {i: membership for i, membership in enumerate(partition.membership)}
                
            except ImportError:
                logger.warning("leidenalg or igraph not installed. Falling back to k-means.")
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_np)
                
                subgroups = {i: label for i, label in enumerate(labels)}
                
        except ImportError:
            logger.warning("networkx not installed. Falling back to k-means.")
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_np)
            
            subgroups = {i: label for i, label in enumerate(labels)}
            
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_np)
        
        subgroups = {i: label for i, label in enumerate(labels)}
        
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hc.fit_predict(embeddings_np)
        
        subgroups = {i: label for i, label in enumerate(labels)}
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Count subgroup sizes
    subgroup_counts = {}
    for label in subgroups.values():
        if label not in subgroup_counts:
            subgroup_counts[label] = 0
        subgroup_counts[label] += 1
    
    logger.info(f"Discovered {len(subgroup_counts)} subgroups with sizes: {subgroup_counts}")
    
    return subgroups