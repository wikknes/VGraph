# Cgraph: Getting Started

Welcome to Cgraph, a comprehensive framework for multi-omics data integration using graph neural networks. This guide will walk you through setting up and running the Cgraph pipeline from installation to analyzing results.

## Table of Contents

1. [Installation](#installation)
   - [Setting up the Environment](#setting-up-the-environment)
   - [Installing Dependencies](#installing-dependencies)
2. [Data Preparation](#data-preparation)
   - [Data Format Requirements](#data-format-requirements)
   - [Using Pseudo Data](#using-pseudo-data)
3. [Running the Pipeline](#running-the-pipeline)
   - [Basic Usage](#basic-usage)
   - [Advanced Options](#advanced-options)
4. [Customization](#customization)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Adding New Modalities](#adding-new-modalities)
   - [Modifying the Graph Structure](#modifying-the-graph-structure)
5. [Interpreting Results](#interpreting-results)
   - [Visualizing Embeddings](#visualizing-embeddings)
   - [Modality Importance](#modality-importance)
   - [Cross-Modality Correlations](#cross-modality-correlations)
6. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
   - [Performance Optimization](#performance-optimization)

## Installation

### Setting up the Environment

Cgraph requires Python 3.9 or higher and is compatible with both CPU and GPU environments. For optimal performance, a CUDA-compatible GPU is recommended.

```bash
# Clone the repository
git clone https://github.com/yourusername/Cgraph.git
cd Cgraph

# Create a conda environment
conda create -n cgraph python=3.9
conda activate cgraph
```

### Installing Dependencies

All required packages are listed in the `requirements.txt` file:

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended)
# Check PyTorch website for the correct CUDA version command:
# https://pytorch.org/get-started/locally/
# Example for CUDA 11.6:
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Data Preparation

### Data Format Requirements

Cgraph expects data in CSV format with the following structure:

- Each modality should be in a separate CSV file
- Files should include a column with participant/sample IDs (default expected name: `lab_ID`)
- Features should be in columns with each row representing a participant/sample
- Missing values should be represented as empty cells or NaN

Example data structure for metabolomics data:

```
lab_ID,metabolite_1,metabolite_2,metabolite_3,...
SUBJ0001,0.5,1.2,0.9,...
SUBJ0002,0.7,NA,1.1,...
```

### Using Pseudo Data

Cgraph includes a built-in pseudo data generator for testing and development:

```bash
# Generate standard pseudo data
python -m Cgraph.examples.run_pipeline --generate_data --data_dir=sample_data

# Generate high-accuracy data with clear patterns
python -m Cgraph.examples.run_pipeline --generate_data --high_accuracy --data_dir=sample_data

# Customize data generation
python -m Cgraph.examples.run_pipeline --generate_data --data_dir=sample_data \
  --n_participants=200 --n_features=100 --n_clusters=4 --missing_rate=0.1
```

You can also programmatically generate data:

```python
from Cgraph.src.data_processing.pseudo_data_generator import generate_pseudo_omics_data

# Generate pseudo data with custom parameters
metadata = generate_pseudo_omics_data(
    output_dir="custom_data",
    n_participants=300,
    n_clusters=5,
    feature_counts={
        'metabolomics': 150,
        'proteomics': 200,
        'biochemistry': 80,
        'lifestyle': 40,
        'lipidomics': 120
    },
    missing_rates={
        'metabolomics': 0.15,
        'proteomics': 0.20,
        'biochemistry': 0.10,
        'lifestyle': 0.05,
        'lipidomics': 0.25
    }
)
```

## Running the Pipeline

### Basic Usage

Running the full pipeline is straightforward:

```python
from Cgraph.pipeline import MultiOmicsIntegration

# Initialize pipeline
pipeline = MultiOmicsIntegration()

# Load data
pipeline.load_data(
    data_dir="path/to/data",
    metabolomics_file="metabolomics.csv",
    proteomics_file="proteomics.csv",
    biochemistry_file="biochemistry.csv",
    lifestyle_file="lifestyle.csv",
    lipidomics_file="lipidomics.csv"
)

# Run pipeline
embeddings = pipeline.run()

# Save results
pipeline.save_results("results/")
```

Using the command-line script:

```bash
# Run the pipeline on existing data
python -m Cgraph.examples.run_pipeline --data_dir=path/to/data --output_dir=results
```

### Advanced Options

The pipeline has multiple configuration options:

```python
pipeline = MultiOmicsIntegration(
    # Model architecture
    embedding_dim=256,      # Dimension of embeddings
    n_heads=4,              # Number of attention heads
    n_layers=2,             # Number of transformer layers
    
    # Graph construction
    correlation_threshold=0.5,  # For feature-feature edges
    n_neighbors=15,             # For graph imputation
    
    # Training parameters
    use_mini_batch=True,    # Use mini-batch training
    batch_size=32,          # Batch size
    use_amp=True,           # Use automatic mixed precision
    early_stopping=True,    # Use early stopping
    patience=10,            # Patience for early stopping
    
    # Hardware
    device="cuda"           # Use GPU if available
)
```

## Customization

### Hyperparameter Tuning

To find optimal hyperparameters:

```python
from Cgraph.pipeline import MultiOmicsIntegration

# Define parameter grid
param_grid = {
    'embedding_dim': [128, 256, 512],
    'n_heads': [2, 4, 8],
    'n_layers': [1, 2, 3],
    'correlation_threshold': [0.3, 0.5, 0.7]
}

best_params = {}
best_score = 0

# Simple grid search example
for embedding_dim in param_grid['embedding_dim']:
    for n_heads in param_grid['n_heads']:
        for n_layers in param_grid['n_layers']:
            for corr_threshold in param_grid['correlation_threshold']:
                
                # Initialize pipeline with current params
                pipeline = MultiOmicsIntegration(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    correlation_threshold=corr_threshold
                )
                
                # Load data and run pipeline
                pipeline.load_data(data_dir="path/to/data")
                pipeline.run()
                
                # Evaluate performance (this is a placeholder for your metric)
                # This could be cluster separation, downstream task accuracy, etc.
                score = evaluate_performance(pipeline)
                
                # Update best params if improved
                if score > best_score:
                    best_score = score
                    best_params = {
                        'embedding_dim': embedding_dim,
                        'n_heads': n_heads,
                        'n_layers': n_layers,
                        'correlation_threshold': corr_threshold
                    }

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

### Adding New Modalities

The pipeline is designed to be flexible with the number and types of modalities:

```python
# Adding a new modality (transcriptomics)
pipeline.load_data(
    data_dir="path/to/data",
    metabolomics_file="metabolomics.csv",
    proteomics_file="proteomics.csv",
    biochemistry_file="biochemistry.csv",
    lifestyle_file="lifestyle.csv",
    lipidomics_file="lipidomics.csv",
    # New modality
    transcriptomics_file="transcriptomics.csv"
)
```

To fully support a new modality, you may need to:

1. Update the data loading function in `Cgraph/src/data_processing/data_loader.py`
2. Add modality-specific preprocessing in `Cgraph/src/data_processing/preprocessing.py`
3. Update the graph construction to include the new modality in `Cgraph/src/graph_construction/heterogeneous_graph.py`

### Modifying the Graph Structure

You can customize the graph structure by:

1. Modifying the correlation threshold:
   ```python
   pipeline = MultiOmicsIntegration(correlation_threshold=0.7)  # Higher = fewer edges
   ```

2. Adding prior knowledge edges:
   ```python
   from Cgraph.src.graph_construction.heterogeneous_graph import add_prior_knowledge_edges
   
   # After building the graph
   interactions_df = pd.read_csv("pathway_interactions.csv")
   graph = add_prior_knowledge_edges(
       pipeline.multi_omics_graph,
       interactions_df,
       source_col="source_feature",
       target_col="target_feature",
       modality_col="modality",
       weight_col="confidence",
       edge_type="interacts_with"
   )
   
   # Use the custom graph
   embeddings = pipeline.run(graph=graph)
   ```

## Interpreting Results

### Visualizing Embeddings

Cgraph provides built-in visualization tools:

```python
from Cgraph.src.visualization.embedding_viz import visualize_participant_embeddings

# After running the pipeline
visualize_participant_embeddings(
    pipeline.embeddings['participant'],
    participant_ids=pipeline.participant_ids,
    method='umap',  # or 'tsne', 'pca'
    output_file="participant_embeddings.png"
)

# Visualize with cluster coloring (if cluster labels are available)
visualize_participant_embeddings(
    pipeline.embeddings['participant'],
    participant_ids=pipeline.participant_ids,
    method='umap',
    cluster_labels=metadata['cluster_assignments'],  # From pseudo data generator
    output_file="participant_clusters.png"
)
```

### Modality Importance

Evaluate the contribution of each modality:

```python
# After running the pipeline
modality_importance = pipeline.evaluate_modality_importance()

# Plot importance scores
from Cgraph.src.visualization.embedding_viz import plot_modality_importance
plot_modality_importance(modality_importance, "modality_importance.png")

# Print scores
for modality, score in sorted(modality_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{modality}: {score:.4f}")
```

### Cross-Modality Correlations

Discover correlations between features from different modalities:

```python
from Cgraph.src.visualization.embedding_viz import discover_cross_modality_correlations

# After running the pipeline
correlations = discover_cross_modality_correlations(
    pipeline,
    source_modality='metabolomics',
    target_modality='proteomics',
    correlation_threshold=0.7,
    output_file="cross_modality_correlations.csv"
)

# Print top correlations
for corr in sorted(correlations, key=lambda x: x['score'], reverse=True)[:10]:
    print(f"{corr['source_feature']} ({corr['source_modality']}) <--> "
          f"{corr['target_feature']} ({corr['target_modality']}): {corr['score']:.3f}")
```

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Try reducing batch size or embedding dimensions
- Enable mini-batch training: `use_mini_batch=True`
- Process data on CPU first: `device="cpu"`

**Slow Training**
- Enable automatic mixed precision: `use_amp=True`
- Reduce graph complexity: increase `correlation_threshold`
- Reduce number of features with preprocessing

**Poor Results**
- Try increasing `embedding_dim` and `n_heads`
- Check for data quality issues (too many missing values)
- Ensure modalities have enough overlapping participants

### Performance Optimization

For large datasets:
- Use mini-batch training: `use_mini_batch=True, batch_size=32`
- Enable automatic mixed precision: `use_amp=True`
- Consider feature selection to reduce dimensionality
- Use neighbor sampling for very large graphs

GPU acceleration:
- Ensure PyTorch and PyTorch Geometric are installed with CUDA support
- Monitor GPU memory usage with `nvidia-smi`
- Set device explicitly: `device="cuda:0"` for multi-GPU systems

## Additional Resources

- [Project Repository](https://github.com/yourusername/Cgraph)
- [Documentation](https://github.com/yourusername/Cgraph/docs)
- [Issue Tracker](https://github.com/yourusername/Cgraph/issues)
- [Related Publications](#)

---

For more details, refer to the API documentation in the code docstrings and the [cookbook.md](cookbook.md) file for specific use cases and recipes.