# Multi-Omics Integration Pipeline Cookbook

This cookbook provides practical recipes and examples for using the Multi-Omics Integration Pipeline based on graph neural networks and inspired by Large Language Model (LLM) architectures.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preparation](#data-preparation)
4. [Common Analysis Workflows](#common-analysis-workflows)
5. [Advanced Integration Recipes](#advanced-integration-recipes)
6. [Troubleshooting](#troubleshooting)

## Introduction

The Multi-Omics Integration Pipeline enables researchers to integrate different types of omics data (metabolomics, proteomics, biochemistry, lifestyle data, and lipidomics) using heterogeneous graph neural networks. This approach is inspired by the architecture of Large Language Models (LLMs), allowing us to capture complex relationships between different biological entities and data modalities.

Key features:
- Integration of up to 5 different omics modalities
- Handling of missing data through graph-based imputation
- Heterogeneous graph construction capturing multi-modal relationships
- LLM-inspired graph transformer model for unified representation learning
- Visualization and downstream analysis tools

## Setup and Installation

### Environment Setup

```bash
# Create and activate conda environment
conda create -n cgraph python=3.9
conda activate cgraph

# Install core dependencies
pip install -r requirements.txt

# Optional: Install GPU support for PyTorch
# Check https://pytorch.org/get-started/locally/ for specific commands
```

### Quick Start

```python
from cgraph.pipeline import MultiOmicsIntegration

# Initialize pipeline
pipeline = MultiOmicsIntegration()

# Load data
pipeline.load_data(
    metabolomics_file="data/metabolomics.csv",
    proteomics_file="data/proteomics.csv",
    biochemistry_file="data/biochemistry.csv",
    lifestyle_file="data/lifestyle.csv",
    lipidomics_file="data/lipidomics.csv"
)

# Run full pipeline
embeddings = pipeline.run()

# Save results
pipeline.save_results("results/model_output")
```

## Data Preparation

### Required Data Format

Each omics dataset should be provided in CSV format with the following structure:
- First column should contain participant IDs (lab_ID)
- Remaining columns should contain features (e.g., metabolite measurements)
- Missing values can be represented as NaN or empty cells

Example:

```
lab_ID,feature1,feature2,feature3,...
SUBJ001,0.5,1.2,0.8,...
SUBJ002,0.7,NaN,1.1,...
SUBJ003,0.4,0.9,0.7,...
```

### Data Preprocessing Recommendations

1. **Feature selection**: When dealing with high-dimensional data, consider:
   - Removing features with high missing rates (>50%)
   - Filtering low-variance features
   - Applying domain knowledge to select relevant biomarkers

2. **Quality control**:
   - Check for batch effects and correct if necessary
   - Identify and handle outliers
   - Log-transform skewed distributions

```python
# Example: Preprocessing metabolomics data
import pandas as pd
from cgraph.data_processing import preprocess_omics_data

# Load raw data
metabolomics_df = pd.read_csv("data/raw/metabolomics.csv")

# Preprocess
processed_df = preprocess_omics_data(
    metabolomics_df,
    id_col="Subject_ID",  # Will be renamed to lab_ID
    missing_threshold=0.3,  # Remove features with >30% missing
    log_transform=True,
    scaling="z-score"
)

# Save processed data
processed_df.to_csv("data/processed/metabolomics.csv", index=False)
```

## Common Analysis Workflows

### 1. Participant Similarity Analysis

This recipe shows how to identify similar participants based on multi-omics data:

```python
from cgraph.pipeline import MultiOmicsIntegration
from cgraph.visualization import visualize_participant_similarity

# Run pipeline
pipeline = MultiOmicsIntegration()
pipeline.load_data(...)
embeddings = pipeline.run()

# Identify similar participants
similar_participants = pipeline.find_similar_participants(
    participant_id="SUBJ001",
    n_neighbors=10
)
print(similar_participants)

# Visualize participant similarity as a network
visualize_participant_similarity(
    embeddings=embeddings["participant"],
    participant_ids=pipeline.participant_ids,
    similarity_threshold=0.7,
    output_file="results/participant_similarity_network.png"
)
```

### 2. Modality Contribution Analysis

This recipe analyzes how each omics modality contributes to the integrated model:

```python
from cgraph.pipeline import MultiOmicsIntegration
from cgraph.visualization import plot_modality_importance

# Run pipeline
pipeline = MultiOmicsIntegration()
pipeline.load_data(...)
pipeline.run()

# Evaluate modality importance
modality_importance = pipeline.evaluate_modality_importance()

# Plot results
plot_modality_importance(
    modality_importance,
    output_file="results/modality_importance.png"
)
```

### 3. Feature Relationship Discovery

This recipe identifies cross-modality feature relationships:

```python
from cgraph.pipeline import MultiOmicsIntegration
from cgraph.analysis import discover_cross_modality_correlations

# Run pipeline
pipeline = MultiOmicsIntegration()
pipeline.load_data(...)
pipeline.run()

# Discover relationships between metabolites and proteins
relationships = discover_cross_modality_correlations(
    pipeline=pipeline,
    source_modality="metabolomics",
    target_modality="proteomics",
    correlation_threshold=0.5
)

# Output top relationships
for rel in relationships[:10]:
    print(f"{rel['source_feature']} ({rel['source_modality']}) <--> "
          f"{rel['target_feature']} ({rel['target_modality']}): {rel['score']:.3f}")
```

## Advanced Integration Recipes

### 1. Custom Graph Structure

Customize the graph structure to incorporate prior knowledge:

```python
from cgraph.pipeline import MultiOmicsIntegration
from cgraph.graph_construction import add_prior_knowledge_edges

# Initialize pipeline
pipeline = MultiOmicsIntegration()
pipeline.load_data(...)

# Build the initial graph
graph = pipeline.build_graph()

# Load prior knowledge (e.g., from a pathway database)
prior_knowledge = pd.read_csv("data/pathway_interactions.csv")

# Add edges based on prior knowledge
graph = add_prior_knowledge_edges(
    graph=graph,
    interactions_df=prior_knowledge,
    source_col="source",
    target_col="target",
    modality_col="modality",
    weight_col="confidence"
)

# Continue with the customized graph
embeddings = pipeline.run(graph=graph)
```

### 2. Transfer Learning Between Cohorts

Apply a model trained on one cohort to another:

```python
from cgraph.pipeline import MultiOmicsIntegration
from cgraph.models import transfer_learned_model

# Train on cohort A
pipeline_A = MultiOmicsIntegration()
pipeline_A.load_data(data_dir="data/cohort_A/")
pipeline_A.run()
pipeline_A.save_model("models/cohort_A_model.pt")

# Transfer to cohort B
pipeline_B = MultiOmicsIntegration()
pipeline_B.load_data(data_dir="data/cohort_B/")

# Apply transfer learning
transfer_learned_model(
    source_pipeline=pipeline_A,
    target_pipeline=pipeline_B,
    freeze_layers=[0],  # Freeze first layer weights
    fine_tune_epochs=50
)

# Run the fine-tuned model on cohort B
embeddings_B = pipeline_B.run()
```

### 3. Subgroup Discovery

Discover subgroups in your cohort based on multi-omics profiles:

```python
from cgraph.pipeline import MultiOmicsIntegration
from cgraph.analysis import discover_subgroups
from cgraph.visualization import visualize_subgroups

# Run pipeline
pipeline = MultiOmicsIntegration()
pipeline.load_data(...)
embeddings = pipeline.run()

# Discover subgroups
subgroups = discover_subgroups(
    embeddings=embeddings["participant"],
    n_clusters=5,
    method="leiden"  # Or "kmeans", "hierarchical", etc.
)

# Get enriched features per subgroup
enrichment = pipeline.compute_subgroup_enrichment(subgroups)

# Visualize subgroups
visualize_subgroups(
    embeddings=embeddings["participant"],
    subgroups=subgroups,
    enrichment=enrichment,
    output_file="results/subgroups_umap.png"
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory errors during graph construction**:
   - Reduce the correlation threshold in graph construction
   - Filter low-information features before building the graph
   - Process data in batches with `batch_size` parameter

2. **Missing data handling**:
   - Increase `n_neighbors` in the imputation step for more stable results
   - For high missingness (>50%), consider excluding problematic modalities
   - Use the `min_participants_per_modality` parameter to enforce minimum coverage

3. **GPU acceleration issues**:
   - Ensure PyTorch with CUDA support is properly installed
   - Set the `device` parameter to explicitly control processing device
   - For very large cohorts, use `use_mini_batch=True` option

### Performance Optimization Tips

- For faster training, reduce `embedding_dim` (default: 256)
- Use `early_stopping=True` to prevent overfitting and reduce training time
- Enable `use_amp=True` for mixed precision training on compatible GPUs

```python
# Example: Optimized pipeline for large datasets
pipeline = MultiOmicsIntegration(
    embedding_dim=128,
    use_mini_batch=True,
    batch_size=512,
    use_amp=True,
    early_stopping=True,
    patience=10
)
```

### Getting Help

For additional support, please:
- Check the detailed documentation in the `docs/` directory
- Search for similar issues in the GitHub repository
- Contact the developers via GitHub issues