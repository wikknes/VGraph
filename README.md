# Cgraph: Multi-Omics Integration with Graph Neural Networks

Cgraph is a comprehensive framework for integrating multiple omics modalities (metabolomics, proteomics, biochemistry, lifestyle data, and lipidomics) using heterogeneous graph neural networks inspired by Large Language Model (LLM) architectures.

## Features

- Integration of up to 5 different omics modalities
- Advanced graph-based imputation for handling missing data
- Heterogeneous graph construction capturing relationships across modalities
- LLM-inspired graph transformer model for unified representation learning
- Visualization and analysis tools for interpreting results
- Support for downstream tasks like clustering and correlation analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Cgraph.git
cd Cgraph

# Create and activate conda environment
conda create -n cgraph python=3.9
conda activate cgraph

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU support for PyTorch
# Check https://pytorch.org/get-started/locally/ for specific commands
```

## Quick Start

```python
from Cgraph.pipeline import MultiOmicsIntegration

# Initialize pipeline
pipeline = MultiOmicsIntegration()

# Load data
pipeline.load_data(
    metabolomics_file="data/metabolomics.csv",
    proteomics_file="data/proteomics.csv",
    biochemistry_file="data/lifestyle.csv",
    lifestyle_file="data/lifestyle.csv",
    lipidomics_file="data/lipidomics.csv"
)

# Run full pipeline
embeddings = pipeline.run()

# Evaluate modality importance
modality_importance = pipeline.evaluate_modality_importance()

# Save results
pipeline.save_results("results/model_output")
```

## Example

The repository includes an example script that demonstrates the pipeline on sample data:

```bash
# Generate sample data and run the pipeline
python Cgraph/examples/run_pipeline.py --generate_data --data_dir=sample_data --output_dir=results
```

## Directory Structure

```
Cgraph/
├── Cgraph/                 # Main package
│   ├── src/                # Source code
│   │   ├── data_processing/    # Data loading and preprocessing
│   │   ├── imputation/         # Graph-based imputation
│   │   ├── graph_construction/ # Heterogeneous graph construction
│   │   ├── models/             # ML models (HGT, transformers)
│   │   └── visualization/      # Visualization utilities
│   ├── examples/           # Example scripts
│   └── pipeline.py         # Main pipeline script
├── cookbook.md             # Detailed examples and recipes
├── data/                   # Data directory
├── notebooks/              # Jupyter notebooks with examples
├── results/                # Results directory
└── tests/                  # Unit tests
```

## Documentation

- **cookbook.md**: Contains detailed examples, use cases, and recipes for common tasks
- **Pipeline API**: Available in docstrings and Jupyter notebooks

## Requirements

- Python 3.9+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- Pandas, NumPy, Scikit-learn
- UMAP, Matplotlib, Seaborn (for visualization)

## Citation

If you use Cgraph in your research, please cite our work:

```
@software{cgraph2023,
  author = {Your Name},
  title = {Cgraph: Multi-Omics Integration with Graph Neural Networks},
  year = {2023},
  url = {https://github.com/yourusername/Cgraph}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.