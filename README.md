# DAE for Pharmaceutical Formulation Data Imputation

This pipeline reproduces the results from the paper:
**"Machine Learning Recovers Corrupted Pharmaceutical 3D Printing Formulation Data"**

## Overview

The pipeline implements **Denoising Autoencoders (DAEs)** to impute missing values in pharmaceutical formulation data. The dataset contains 1,570+ formulations across 382 ingredients with ~99% sparsity (zeros for unused ingredients).

### Key Features

- **MPS Support**: Automatically uses Apple Silicon GPU (MPS) for training on Mac
- **Reproducible**: Uses fixed random seeds (42, 50, 100) for statistical robustness
- **Comprehensive**: Tests 243 DAE experiments and 90 KNN experiments
- **Publication-Ready**: Generates plots matching the paper figures
- **Masked Loss**: Loss computed only on artificially corrupted values
- **Multiple Baselines**: Includes KNN and zero imputation for comparison
- **Parallel Execution**: KNN experiments run in parallel for efficiency

### DAE Architecture

The Denoising Autoencoder uses an overcomplete architecture:
```
Input (382 dims)
  → FC + BatchNorm + LeakyReLU (hidden_dim)
  → FC + BatchNorm + LeakyReLU (latent_dim)
  → FC + Sigmoid (382 dims)
```

Key design choices:
- **Overcomplete**: `hidden_dim = latent_dim = neuron_size` (256/512/1024)
- **Denoising**: Gaussian noise (σ=0.1) added during training
- **Masked loss**: Only corrupted values contribute to loss (not entire reconstruction)

### Data Corruption Strategy

Training uses a two-step corruption process:
1. **Masking**: Randomly select values based on missingness rate (1%, 5%, 10%)
2. **Zeroing + Noise**: Set masked values to zero, then add Gaussian noise (σ=0.1) to all values

The model learns to denoise and impute the masked values while ignoring the ~99% of naturally-occurring zeros (unused ingredients).

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Quick Test (Recommended First)

Test the DAE pipeline with reduced parameters (approximately 5 minutes):

```bash
python main.py --quick-test
```

This runs 1 experiment with:
- 1% missingness
- Learning rate: 10⁻³
- 256 neurons
- 100 epochs
- 1 seed

Test the KNN pipeline (approximately 10 seconds):

```bash
python knn_main.py --quick-test
```

### Run Full Experiments

**DAE Experiments** - Reproduce all paper results (approximately 6-8 hours on Apple Silicon):

```bash
python main.py
```

This runs **243 experiments**:
- 3 missingness rates (1%, 5%, 10%)
- 3 learning rates (10⁻¹, 10⁻³, 10⁻⁵)
- 3 neuron sizes (256, 512, 1024)
- 4 epoch settings (100, 500, 1000, 1200)
- 3 seeds per experiment

**KNN Experiments** - Classical ML baseline (approximately 15-30 minutes):

```bash
python knn_main.py
```

This runs **90 experiments** in parallel:
- 3 missingness rates (1%, 5%, 10%)
- 5 n-neighbors (3, 5, 10, 20, 50)
- 2 weights (uniform, distance)
- 3 metrics (euclidean, manhattan, cosine)
- 3 seeds per experiment

### Generate Method Comparisons

After running both DAE and KNN experiments, generate comparison plots:

```bash
python -c "from src.comparison.plots import generate_all_comparisons; generate_all_comparisons()"
```

This creates:
- R² comparison bar chart (DAE vs KNN vs Zero)
- Performance vs. time scatter plots
- Detailed comparison table

### Custom Experiments

Run specific configurations:

**DAE:**
```bash
# Test only 1% missingness
python main.py --missingness 0.01

# Test specific learning rates
python main.py --learning-rates 0.001 0.00001

# Test specific neuron sizes and epochs
python main.py --neuron-sizes 512 1024 --epochs 500 1000

# Use different seeds
python main.py --seeds 1 2 3 4 5
```

**KNN:**
```bash
# Test specific K values
python knn_main.py --k-neighbors 5 10 20

# Test specific weights and metrics
python knn_main.py --weights uniform distance --metrics euclidean

# Test specific missingness rates
python knn_main.py --missingness 0.01 0.05
```

### Generate Plots Only

If you already ran experiments and just want to regenerate plots:

```bash
python main.py --skip-training
```

## Project Structure

```
DAE/
├── data/
│   └── material_name_smilesRemoved.csv    # Formulation dataset
├── src/
│   ├── common/                            # Shared utilities
│   │   ├── data_preprocessing.py          # Data loading, normalization, corruption
│   │   └── visualization.py               # Common plotting utilities
│   ├── dae/                               # Denoising Autoencoder
│   │   ├── model.py                       # DAE architecture & device selection
│   │   ├── train.py                       # Training loop with masked loss
│   │   ├── evaluate.py                    # Metrics computation
│   │   └── plots.py                       # DAE-specific visualizations
│   ├── knn/                               # K-Nearest Neighbors baseline
│   │   ├── imputation.py                  # KNN imputer
│   │   ├── evaluate.py                    # KNN metrics & timing
│   │   └── plots.py                       # KNN visualizations
│   ├── baselines/                         # Additional baseline methods
│   │   ├── zero_imputer.py                # Naive zero-filling baseline
│   │   ├── evaluate.py                    # Baseline metrics
│   │   └── plots.py                       # Baseline visualizations
│   └── comparison/                        # Cross-method comparisons
│       └── plots.py                       # DAE vs KNN vs Zero comparisons
├── results/
│   ├── dae/                               # DAE outputs
│   │   ├── models/                        # Trained model checkpoints
│   │   ├── metrics/                       # Metrics & predictions
│   │   ├── plots/                         # DAE figures
│   │   └── summary.json                   # All DAE results
│   ├── knn/                               # KNN outputs
│   │   ├── metrics/                       # KNN metrics
│   │   ├── predictions/                   # KNN predictions
│   │   ├── plots/                         # KNN figures
│   │   └── summary.json                   # All KNN results
│   ├── baselines/                         # Baseline outputs
│   └── comparisons/                       # Method comparison outputs
│       ├── method_comparison.png
│       ├── performance_vs_time.png
│       └── comparison_table.txt
├── requirements.txt                       # Python dependencies
├── main.py                                # DAE pipeline orchestrator
├── knn_main.py                            # KNN baseline pipeline
├── CLAUDE.md                              # Developer documentation
└── README.md                              # This file
```

## Output Files

### DAE Models

Saved to `results/dae/models/`:
- Filename format: `miss{rate}_lr{lr}_n{neurons}_ep{epochs}_seed{seed}.pt`
- Example: `miss0.01_lr0.001_n512_ep1000_seed42.pt`

### DAE Metrics

Saved to `results/dae/metrics/`:
- Loss histories: `*_loss.json`
- Aggregated metrics: `*_metrics.json`
- Predictions: `*_predictions.npz`

### KNN Results

Saved to `results/knn/`:
- Metrics: `metrics/miss{rate}_k{neighbors}_{weights}_{metric}_metrics.json`
- Predictions: `predictions/miss{rate}_k{neighbors}_{weights}_{metric}_predictions.npz`

### Plots

**DAE plots** saved to `results/dae/plots/`:
- Loss curves: `loss_curves_{pct}pct.png`
- R² bar plots: `r2_bars_{pct}pct.png`
- Predicted vs Truth: `pred_vs_truth_1pct.png`

**KNN plots** saved to `results/knn/plots/`:
- K-neighbor comparison: `k_comparison_{pct}pct.png`
- Weights comparison: `weights_comparison_{pct}pct.png`

**Method comparisons** saved to `results/comparisons/`:
- Bar chart: `method_comparison.png`
- Performance vs time: `performance_vs_time.png`
- Comparison table: `comparison_table.txt`

## Key Results

The paper reports these R² scores for DAE imputation:

| Missing Data | Best R² (mean ± std) | Configuration |
|--------------|----------------------|---------------|
| 1%           | 0.94 ± 0.03         | 1024 neurons, 1200 epochs, LR=10⁻³ |
| 5%           | 0.48 ± 0.09         | 256 neurons, lower epochs, LR=10⁻³ |
| 10%          | 0.37 ± 0.05         | 256 neurons, lower epochs, LR=10⁻³ |

### Key Findings

1. **Learning rate has the strongest effect** on DAE performance
   - 10⁻³ performs best across all missingness levels
   - 10⁻¹ and 10⁻⁵ produce negative R² scores (worse than baseline)

2. **Larger models work better for low missingness** (1%)
   - 1024 neurons optimal for 1% missing

3. **Smaller models work better for high missingness** (5-10%)
   - 256 neurons optimal for higher missingness
   - Suggests smaller models generalize better with less signal

4. **Method comparison**
   - Run both DAE and KNN experiments to compare neural network vs. classical ML approaches
   - Use `generate_all_comparisons()` to create side-by-side performance analysis

## Hardware Requirements

- **CPU**: Any modern CPU (2+ cores recommended)
- **GPU**: Optional but recommended
  - Apple Silicon (M1/M2/M3): Automatically uses MPS
  - NVIDIA: Automatically uses CUDA if available
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~500MB for models and results

## Estimated Runtime

On Apple M1 Pro:
- Quick test: ~5 minutes
- Full pipeline (243 experiments): ~6-8 hours

On CPU only:
- Quick test: ~15 minutes
- Full pipeline: ~24-48 hours

## Module Usage

The modular structure allows independent use of each component:

### Data Preprocessing

```python
from src.common.data_preprocessing import FormulationDataPreprocessor

preprocessor = FormulationDataPreprocessor(
    data_path='data/material_name_smilesRemoved.csv',
    metadata_cols=6
)
preprocessor.load_data()
preprocessor.normalize_data()

# Corrupt 1% of data
original, corrupted, mask = preprocessor.prepare_data(
    missingness_rate=0.01,
    noise_std=0.1,
    seed=42
)
```

### DAE Model Creation

```python
from src.dae.model import create_dae, get_device

device = get_device()
model = create_dae(input_dim=382, neuron_size=512, device=device)
```

### DAE Training

```python
from src.dae.train import train_dae

trained_model, loss_history = train_dae(
    model=model,
    original_data=original,
    corrupted_data=corrupted,
    mask=mask,
    device=device,
    learning_rate=1e-3,
    num_epochs=1000
)
```

### DAE Evaluation

```python
from src.dae.evaluate import evaluate_dae

predictions, metrics = evaluate_dae(
    model=trained_model,
    original_data=original,
    corrupted_data=corrupted,
    mask=mask,
    device=device,
    noise_std=0.1
)

print(f"R²: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### KNN Imputation

```python
from src.knn.imputation import create_knn_imputer

knn_imputer = create_knn_imputer(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)

# Convert to numpy
original_np = original.numpy()
corrupted_np = corrupted.numpy()
mask_np = mask.numpy()

# Impute
imputed_data, metrics = knn_imputer.impute(original_np, corrupted_np, mask_np)
```

### Generating Comparison Plots

```python
from src.comparison.plots import generate_all_comparisons

generate_all_comparisons(
    dae_results_dir='results/dae',
    knn_results_dir='results/knn',
    zero_results_dir='results/baselines',
    output_dir='results/comparisons'
)
```

## Contact

For questions or issues:
- Corresponding Author: m.elbadawi@qmul.ac.uk

## Troubleshooting

### MPS Not Available

If you get "MPS not available" on Mac:
- Update to macOS 12.3+ and PyTorch 2.0+
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory

If training crashes with OOM:
1. Reduce neuron sizes: `--neuron-sizes 256`
2. Train on CPU (slower): Edit `src/dae/model.py` to force CPU

### Missing Data File

Ensure `data/material_name_smilesRemoved.csv` exists:
```bash
ls data/material_name_smilesRemoved.csv
```

## Acknowledgments

- Queen Mary University of London
- UCL School of Pharmacy
