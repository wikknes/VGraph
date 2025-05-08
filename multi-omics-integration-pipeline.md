# Multi-Omics Integration Pipeline with Graph Neural Networks: LLM-Inspired Approach

This document outlines a comprehensive framework for integrating metabolomics, proteomics, biochemistry, lifestyle data, and lipidomics from 10,000 participants using heterogeneous graph neural networks inspired by Large Language Model (LLM) architectures. The pipeline handles participants with incomplete modality coverage through advanced graph-based imputation techniques.

## 1. Project Setup & Dependencies

### Required Libraries
```bash
# Core data processing
pip install pandas numpy scikit-learn tqdm 

# Graph ML
pip install torch torch_geometric torch_sparse torch_scatter networkx

# Multi-omics imputation 
pip install snfpy tensorly

# Visualization
pip install matplotlib seaborn plotly umap-learn

# Optional: MOFA2 for alternative imputation
# Requires R with BiocManager and MOFA2 installed
pip install rpy2
```

### Compute Requirements
- **RAM**: Minimum 16GB, recommended 32GB+ for full dataset
- **GPU**: Optional but recommended for GNN training (8GB+ VRAM)
- **Storage**: ~20GB for processed data and model checkpoints

## 2. Data Inventory & Harmonization

### 2.1 Create Master Index
```python
import pandas as pd
import numpy as np
from collections import defaultdict

# Dictionary to track modality availability per participant
modality_presence = defaultdict(dict)

# Load and inventory all CSV files
modalities = {}
modality_names = ['metabolomics', 'proteomics', 'biochemistry', 'lifestyle', 'lipidomics']
all_lab_ids = set()

for modality in modality_names:
    try:
        df = pd.read_csv(f"{modality}.csv")
        # Ensure lab_ID column exists and is correctly named
        id_col = [col for col in df.columns if col.lower() in ['lab_id', 'labid', 'id']][0]
        if id_col != 'lab_ID':
            df.rename(columns={id_col: 'lab_ID'}, inplace=True)
            
        # Track which participants have which modalities
        for lab_id in df['lab_ID']:
            modality_presence[lab_id][modality] = True
            all_lab_ids.add(lab_id)
            
        modalities[modality] = df
        print(f"Loaded {modality}: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading {modality}: {e}")

# Create summary of modality availability
availability_df = pd.DataFrame(index=sorted(all_lab_ids), columns=modality_names)
for lab_id in all_lab_ids:
    for modality in modality_names:
        availability_df.loc[lab_id, modality] = modality in modality_presence[lab_id]

print(f"\nTotal unique participants: {len(all_lab_ids)}")
print(f"Participants with all modalities: {availability_df.all(axis=1).sum()}")

# Save availability matrix
availability_df.to_csv("modality_availability.csv")
```

### 2.2 Standardize Features

```python
def standardize_features(modalities):
    """Standardize features across all modalities."""
    for modality, df in modalities.items():
        # Set lab_ID as index
        if 'lab_ID' in df.columns:
            df.set_index('lab_ID', inplace=True)
            
        # Handle different modalities appropriately
        if modality == 'lifestyle':
            # One-hot encode categorical variables
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
        else:
            # For omics data: log transform positive values, then standardize
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                # Apply log transform to positive values (with offset for zeros)
                mask = df[col] > 0
                if mask.any():
                    min_positive = df.loc[mask, col].min() / 10
                    df.loc[mask, col] = np.log1p(df.loc[mask, col] - min_positive)
                
                # Z-score normalization
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        modalities[modality] = df
    
    return modalities

# Standardize features across all modalities
modalities = standardize_features(modalities)
```

## 3. Missing Data Analysis & Imputation

### 3.1 Visualize Missing Data Patterns

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create missingness heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(availability_df, cmap='viridis', cbar_kws={'label': 'Data Available'})
plt.title('Data Availability Across Modalities')
plt.tight_layout()
plt.savefig('missingness_heatmap.png')

# Calculate and plot missingness percentages by modality
missingness = 1 - availability_df.mean()
plt.figure(figsize=(10, 6))
missingness.plot(kind='bar')
plt.title('Percentage of Missing Data by Modality')
plt.ylabel('Missing Fraction')
plt.tight_layout()
plt.savefig('missingness_by_modality.png')
```

### 3.2 Graph-Based Imputation

```python
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class GRAPEImputer:
    """Graph-based imputation inspired by GRAPE (Graph-Based Population-Level Imputation)"""
    
    def __init__(self, n_neighbors=10, embedding_dim=128):
        self.n_neighbors = n_neighbors
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_participant_similarity_graph(self, feature_matrix):
        """Create a similarity graph based on available features"""
        # Use only complete cases for KNN
        complete_mask = ~np.isnan(feature_matrix).any(axis=1)
        complete_indices = np.where(complete_mask)[0]
        
        if len(complete_indices) < self.n_neighbors:
            raise ValueError(f"Not enough complete cases for KNN (need {self.n_neighbors}, have {len(complete_indices)})")
            
        # Fit KNN on complete cases
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
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
        for i in range(len(feature_matrix)):
            if i not in complete_indices:
                # Create a temporary version of this row with NaNs replaced by means
                temp_row = feature_matrix[i].copy()
                col_means = np.nanmean(feature_matrix, axis=0)
                nan_mask = np.isnan(temp_row)
                temp_row[nan_mask] = col_means[nan_mask]
                
                # Find nearest neighbors among complete cases
                distances, indices = knn.kneighbors([temp_row])
                for neighbor_idx in indices[0]:
                    edge_list.append((i, complete_indices[neighbor_idx]))
                    
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def build_imputation_model(self, input_dim):
        """Build a GNN model for imputation"""
        class GNNImputer(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(GNNImputer, self).__init__()
                self.conv1 = SAGEConv(input_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, hidden_dim)
                self.conv3 = SAGEConv(hidden_dim, output_dim)
                
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.2, training=self.training)
                x = F.relu(self.conv2(x, edge_index))
                x = self.conv3(x, edge_index)
                return x
            
        return GNNImputer(input_dim, self.embedding_dim, input_dim).to(self.device)
    
    def impute(self, data_df):
        """Impute missing values using graph neural network"""
        # Convert DataFrame to numpy array
        feature_matrix = data_df.values.astype(np.float32)
        
        # Create mask of missing values
        missing_mask = np.isnan(feature_matrix)
        
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
        edge_index = self.create_participant_similarity_graph(imputed_matrix)
        
        # Initialize and train GNN model
        model = self.build_imputation_model(feature_matrix.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to PyTorch tensors
        x = torch.tensor(imputed_matrix, dtype=torch.float).to(self.device)
        edge_index = edge_index.to(self.device)
        mask = torch.tensor(~missing_mask, dtype=torch.bool).to(self.device)
        
        # Train model
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = model(x, edge_index)
            
            # Only compute loss on observed values
            loss = F.mse_loss(out[mask], x[mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
        
        # Use trained model for final imputation
        model.eval()
        with torch.no_grad():
            imputed_features = model(x, edge_index).cpu().numpy()
            
        # Replace only the missing values with imputed ones
        result_matrix = feature_matrix.copy()
        result_matrix[missing_mask] = imputed_features[missing_mask]
        
        return pd.DataFrame(result_matrix, index=data_df.index, columns=data_df.columns)

# Apply imputation to each modality
imputed_modalities = {}
imputer = GRAPEImputer(n_neighbors=15, embedding_dim=256)

for modality, df in modalities.items():
    print(f"Imputing missing values for {modality}...")
    try:
        imputed_df = imputer.impute(df)
        imputed_modalities[modality] = imputed_df
    except Exception as e:
        print(f"Error imputing {modality}, using mean imputation instead: {e}")
        # Fallback to simple imputation
        imputed_df = df.fillna(df.mean())
        imputed_modalities[modality] = imputed_df
```

## 4. Heterogeneous Graph Construction

### 4.1 Create Heterogeneous Graph

```python
import torch
from torch_geometric.data import HeteroData

def create_multi_omics_graph(modalities, availability_df, correlation_threshold=0.5):
    """
    Create a heterogeneous graph with participants and features as nodes
    """
    graph = HeteroData()
    
    # Add participant nodes (all participants)
    all_participants = sorted(availability_df.index)
    n_participants = len(all_participants)
    participant_id_map = {pid: i for i, pid in enumerate(all_participants)}
    
    # Initialize participant features (demographic/clinical if available, otherwise zeros)
    if 'lifestyle' in modalities:
        participant_features = modalities['lifestyle'].reindex(all_participants).fillna(0).values
    else:
        participant_features = np.zeros((n_participants, 5))  # Default features
    
    # Add participant nodes to graph
    graph['participant'].x = torch.tensor(participant_features, dtype=torch.float)
    graph['participant'].lab_ids = all_participants
    
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
        
        # Add participant-to-feature edges (with values as edge attributes)
        edge_indices = []
        edge_attrs = []
        
        for pid in all_participants:
            if pid in df.index:  # Participant has this modality
                p_idx = participant_id_map[pid]
                for feature_name in feature_names:
                    f_idx = feature_id_map[feature_name]
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
        
        # Add feature-to-feature edges (based on correlation)
        if len(df) > 10:  # Need enough samples for meaningful correlation
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
        
        feature_offset += n_features
    
    return graph

# Create heterogeneous graph
multi_omics_graph = create_multi_omics_graph(imputed_modalities, availability_df, correlation_threshold=0.5)

print("Graph statistics:")
for node_type in multi_omics_graph.node_types:
    print(f"  {node_type}: {multi_omics_graph[node_type].x.shape[0]} nodes")
for edge_type in multi_omics_graph.edge_types:
    print(f"  {edge_type}: {multi_omics_graph[edge_type].edge_index.shape[1]} edges")
```

## 5. LLM-Inspired Heterogeneous Graph Transformer (HGT)

### 5.1 Implement HGT Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HeterogeneousGraphTransformer(torch.nn.Module):
    """
    Heterogeneous Graph Transformer with attention mechanisms similar to LLMs
    """
    def __init__(self, graph, hidden_channels=256, out_channels=128, num_heads=4, num_layers=2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = num_heads
        self.num_layers = num_layers
        
        # Get node types and edge types from the graph
        self.node_types = graph.node_types
        self.edge_types = graph.edge_types
        
        # Create node type-specific input projections
        self.input_linears = nn.ModuleDict()
        for node_type in self.node_types:
            input_dim = graph[node_type].x.shape[1]
            self.input_linears[node_type] = Linear(input_dim, hidden_channels)
        
        # Stack of HGT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(
                hidden_channels, 
                hidden_channels, 
                self.node_types,
                self.edge_types, 
                num_heads, 
                group='sum'
            )
            self.convs.append(conv)
        
        # Output layer
        self.output_linear = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        """Forward pass through the heterogeneous graph"""
        # Project each node type to the same dimensionality
        h_dict = {node_type: self.input_linears[node_type](x) for node_type, x in x_dict.items()}
        
        # Pass through HGT layers
        for i in range(self.num_layers):
            h_dict_new = {}
            
            # Apply HGT convolution
            for node_type in h_dict.keys():
                h_dict_new[node_type] = self.convs[i](h_dict, edge_index_dict, node_type)
            
            # Add residual connections and apply non-linearity
            for node_type in h_dict.keys():
                h_dict[node_type] = F.leaky_relu(h_dict[node_type] + h_dict_new[node_type])
        
        # Apply output projection
        out_dict = {node_type: self.output_linear(h) for node_type, h in h_dict.items()}
        
        return out_dict
```

### 5.2 Train Model and Generate Embeddings

```python
def train_hgt_model(graph, num_epochs=100, lr=0.001):
    """Train the HGT model and generate embeddings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move graph to device
    graph = graph.to(device)
    
    # Prepare model
    model = HeterogeneousGraphTransformer(graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create a dictionary for input features
    x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
    
    # Create a dictionary for edge indices
    edge_index_dict = {
        edge_type: graph[edge_type].edge_index
        for edge_type in graph.edge_types
    }
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out_dict = model(x_dict, edge_index_dict)
        
        # Compute reconstruction loss (self-supervision)
        loss = 0
        for edge_type in graph.edge_types:
            src, _, dst = edge_type
            edge_index = edge_index_dict[edge_type]
            edge_attr = graph[edge_type].edge_attr if 'edge_attr' in graph[edge_type] else None
            
            # Get embeddings of source and destination nodes
            src_embeds = out_dict[src][edge_index[0]]
            dst_embeds = out_dict[dst][edge_index[1]]
            
            # Compute similarity between embeddings
            pred = (src_embeds * dst_embeds).sum(dim=1)
            
            if edge_attr is not None:
                # Use edge attributes as targets
                target = edge_attr.view(-1)
                loss += F.mse_loss(pred, target)
            else:
                # Default to link prediction loss
                loss += -torch.log(torch.sigmoid(pred)).mean()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        embeddings_dict = model(x_dict, edge_index_dict)
    
    # Move embeddings back to CPU
    embeddings_dict = {k: v.cpu() for k, v in embeddings_dict.items()}
    
    return model, embeddings_dict

# Train model and generate embeddings
hgt_model, embeddings = train_hgt_model(multi_omics_graph, num_epochs=200)

# Save embeddings for all node types
for node_type, embedding in embeddings.items():
    np.save(f'{node_type}_embeddings.npy', embedding.numpy())
```

## 6. Analysis and Visualization

### 6.1 Participant Embedding Analysis

```python
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Get participant embeddings
participant_embeddings = embeddings['participant'].numpy()

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding_2d = reducer.fit_transform(participant_embeddings)

# Plot embeddings
plt.figure(figsize=(12, 10))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5, alpha=0.7)

# Color points by modality completeness
completeness_score = availability_df.sum(axis=1).values
plt.figure(figsize=(12, 10))
plt.scatter(
    embedding_2d[:, 0], 
    embedding_2d[:, 1], 
    c=completeness_score, 
    cmap='viridis', 
    s=10, 
    alpha=0.8
)
plt.colorbar(label='Number of Available Modalities')
plt.title('UMAP Visualization of Participant Embeddings')
plt.tight_layout()
plt.savefig('participant_embeddings.png')
```

### 6.2 Feature Importance Analysis

```python
def analyze_attention_weights(model, graph):
    """Extract and analyze attention weights from HGT model"""
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
    
    return attention_dict

attention_weights = analyze_attention_weights(hgt_model, multi_omics_graph)

# Analyze most important connections
for att_name, att_weight in attention_weights.items():
    if att_weight.numel() > 0:  # Check if tensor is not empty
        mean_att = att_weight.mean(dim=0)  # Average over all instances
        print(f"\nAttention analysis for {att_name}:")
        print(f"Mean attention: {mean_att}")
        print(f"Max attention: {mean_att.max().item()}")
```

## 7. Modality Importance and Downstream Task Examples

### 7.1 Modality Importance Analysis

```python
def evaluate_modality_importance(model, graph):
    """Evaluate importance of each modality by measuring impact on embeddings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create baseline embeddings with all modalities
    with torch.no_grad():
        x_dict = {node_type: graph[node_type].x.to(device) for node_type in graph.node_types}
        edge_index_dict = {edge_type: graph[edge_type].edge_index.to(device) for edge_type in graph.edge_types}
        baseline_embeddings = model(x_dict, edge_index_dict)['participant'].cpu()
    
    # For each modality, remove it and measure change in embeddings
    modality_importance = {}
    modality_types = [node_type for node_type in graph.node_types if node_type != 'participant']
    
    for modality in modality_types:
        # Create a new edge_index_dict without this modality
        filtered_edge_types = [
            edge_type for edge_type in graph.edge_types 
            if modality not in edge_type
        ]
        filtered_edge_index_dict = {
            edge_type: graph[edge_type].edge_index.to(device)
            for edge_type in filtered_edge_types
        }
        
        # Get embeddings without this modality
        with torch.no_grad():
            modality_removed_embeddings = model(x_dict, filtered_edge_index_dict)['participant'].cpu()
        
        # Measure change in embeddings
        embedding_change = torch.norm(baseline_embeddings - modality_removed_embeddings, dim=1).mean().item()
        modality_importance[modality] = embedding_change
    
    return modality_importance

# Evaluate modality importance
modality_importance = evaluate_modality_importance(hgt_model, multi_omics_graph)
print("\nModality importance scores:")
for modality, score in sorted(modality_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{modality}: {score:.4f}")

# Plot modality importance
plt.figure(figsize=(10, 6))
plt.bar(modality_importance.keys(), modality_importance.values())
plt.title('Modality Importance')
plt.ylabel('Embedding Change')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('modality_importance.png')
```

### 7.2 Downstream Classification Task Example

```python
def prepare_classification_task(embeddings, labels_file='outcome_labels.csv'):
    """Prepare data for a classification task using learned embeddings"""
    try:
        # Load outcome labels (e.g., disease status)
        labels_df = pd.read_csv(labels_file)
        labels_df.set_index('lab_ID', inplace=True)
        
        # Get participant embeddings and IDs
        participant_embeddings = embeddings['participant']
        participant_ids = multi_omics_graph['participant'].lab_ids
        
        # Match embeddings with labels
        common_ids = set(participant_ids).intersection(labels_df.index)
        
        if len(common_ids) == 0:
            print("No common IDs found between embeddings and labels")
            return None, None
            
        # Create arrays for matched embeddings and labels
        X = []
        y = []
        
        for i, lab_id in enumerate(participant_ids):
            if lab_id in common_ids:
                X.append(participant_embeddings[i].numpy())
                y.append(labels_df.loc[lab_id, 'outcome'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Prepared classification data: {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
        
    except Exception as e:
        print(f"Error preparing classification task: {e}")
        print("For demonstration, creating synthetic labels instead")
        
        # Create synthetic binary labels for demonstration
        n_participants = embeddings['participant'].shape[0]
        synthetic_y = np.random.randint(0, 2, size=n_participants)
        return embeddings['participant'].numpy(), synthetic_y

# Prepare data for classification
X, y = prepare_classification_task(embeddings)

if X is not None and y is not None:
    # Split into train/test sets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
```

## 8. Saving and Loading the Pipeline

```python
import pickle
import os

def save_pipeline(model, embeddings, graph, output_dir='model_output'):
    """Save the trained model, embeddings, and graph"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, 'hgt_model.pt'))
    
    # Save embeddings
    for node_type, emb in embeddings.items():
        torch.save(emb, os.path.join(output_dir, f'{node_type}_embeddings.pt'))
    
    # Save graph structure
    torch.save(graph, os.path.join(output_dir, 'multi_omics