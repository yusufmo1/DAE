# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Denoising Autoencoders (DAEs)**, **MissForest**, and other ML methods for imputing missing values in pharmaceutical 3D printing formulation data. The dataset contains 1,570+ formulations across 382 ingredients with ~99% sparsity (most ingredients are unused in each formulation).

Paper: "Machine Learning Recovers Corrupted Pharmaceutical 3D Printing Formulation Data" by Olima Uddin, Yusuf Ali Mohammed, Simon Gaisford, and Moe Elbadawi.

**Dependency Management:** This project uses Poetry for dependency management (migrated from requirements.txt).

## Key Commands

### Run All Experiments (Recommended)

```bash
# Quick test of all methods (~15-20 min total)
poetry run python run_all.py --quick-test

# Full pipeline: DAE + KNN + MissForest + Baselines + Comparisons
poetry run python run_all.py

# Run all but skip specific methods (if already completed)
poetry run python run_all.py --skip-dae
poetry run python run_all.py --skip-knn --skip-missforest --skip-baseline

# Generate comparisons only (requires existing results)
poetry run python run_all.py --comparison-only
```

### Running DAE Experiments

```bash
# Quick test (5 min, recommended first run)
poetry run python run_dae.py --quick-test

# Full DAE experiments (~6-8 hours on Apple Silicon, ~24-48h on CPU)
poetry run python run_dae.py

# Custom configurations
poetry run python run_dae.py --missingness 0.01 --learning-rates 0.001 --neuron-sizes 512 --epochs 1000
poetry run python run_dae.py --seeds 1 2 3 4 5

# Generate plots only (skip training)
poetry run python run_dae.py --skip-training
```

### Running KNN Baseline Experiments

```bash
# Quick test (seconds, no training required)
poetry run python run_knn.py --quick-test

# Full KNN experiments (runs in parallel, ~15-30 min)
poetry run python run_knn.py

# Custom KNN configurations
poetry run python run_knn.py --k-neighbors 5 10 20 --weights uniform distance
poetry run python run_knn.py --metrics euclidean manhattan cosine
poetry run python run_knn.py --missingness 0.01 0.05 0.10

# Run sequentially instead of parallel
poetry run python run_knn.py --no-parallel
```

### Running MissForest Experiments

```bash
# Quick test (~5-10 min, reduced iterations)
poetry run python run_missforest.py --quick-test

# Full MissForest experiments (~4-8 hours, trains 336 RandomForests per iteration)
poetry run python run_missforest.py

# Custom configurations
poetry run python run_missforest.py --missingness 0.01 0.05 0.10
poetry run python run_missforest.py --max-iter 5
poetry run python run_missforest.py --seeds 42 50 100

# Generate plots only (skip training)
poetry run python run_missforest.py --skip-training
```

**Note:** MissForest uses sklearn's `IterativeImputer` with `RandomForestRegressor`. It trains one RandomForest per feature (336 models in this dataset), making it computationally expensive but highly accurate.

### Running Baseline Experiments

```bash
# Quick test (naive zero imputation)
poetry run python run_baseline.py --quick-test

# Full baseline experiments
poetry run python run_baseline.py

# Custom configurations
poetry run python run_baseline.py --missingness 0.01 0.05 0.10
poetry run python run_baseline.py --seeds 42 50 100
```

### Generating Method Comparisons

After running DAE, KNN, MissForest, and baseline experiments:
```bash
poetry run python run_comparison.py
```

This generates:
- `results/comparisons/method_comparison.png` - R² comparison bar chart (DAE vs KNN vs MissForest vs Zero)
- `results/comparisons/performance_vs_time.png` - Performance vs. computational time
- `results/comparisons/comparison_table.txt` - Detailed comparison table

### Installing Dependencies

```bash
# Install dependencies using Poetry
poetry install

# Alternative: activate Poetry shell
poetry shell
python run_all.py --quick-test
```

## Architecture & Data Flow

### Modular Structure

The codebase is organized into functional modules:

```
src/
├── common/              # Shared utilities across all methods
│   ├── data_preprocessing.py    # Data loading, normalization, corruption
│   └── visualization.py         # Common plotting utilities
├── dae/                 # Denoising Autoencoder implementation
│   ├── model.py                 # DAE architecture & device selection
│   ├── train.py                 # Training loop with masked loss
│   ├── evaluate.py              # Metrics computation
│   ├── experiments.py           # DAE orchestration
│   └── plots.py                 # DAE-specific visualizations
├── knn/                 # K-Nearest Neighbors baseline
│   ├── imputation.py            # KNN imputer (sklearn wrapper)
│   ├── evaluate.py              # KNN metrics & timing
│   ├── experiments.py           # KNN orchestration
│   └── plots.py                 # KNN visualizations
├── missforest/          # MissForest (IterativeImputer) method
│   ├── imputation.py            # sklearn IterativeImputer + RandomForest
│   ├── evaluate.py              # MissForest metrics & timing
│   ├── experiments.py           # MissForest orchestration
│   └── plots.py                 # MissForest visualizations
├── baselines/           # Additional baseline methods
│   ├── zero_imputer.py          # Naive zero-filling baseline
│   ├── evaluate.py              # Baseline metrics
│   ├── experiments.py           # Baseline orchestration
│   └── plots.py                 # Baseline visualizations
└── comparison/          # Cross-method comparison tools
    └── plots.py                 # DAE vs KNN vs MissForest vs Zero comparison plots
```

### DAE Pipeline Flow

1. **Data Preprocessing** (`src/common/data_preprocessing.py`)
   - `FormulationDataPreprocessor` class handles all data operations
   - Loads CSV, extracts ingredient columns (skips 6 metadata columns)
   - Normalizes to [0, 1] using MinMaxScaler
   - Creates corruption: randomly masks values → sets to zero → adds Gaussian noise (σ=0.1)

2. **Model Creation** (`src/dae/model.py`)
   - `create_dae()` builds the architecture
   - DAE structure: `Input → [FC+BN+LeakyReLU] → [FC+BN+LeakyReLU] → [FC+Sigmoid] → Output`
   - Both hidden and latent layers use same neuron count (256/512/1024)
   - Overcomplete architecture (latent_dim = hidden_dim = neuron_size)
   - `get_device()` automatically selects CUDA > MPS > CPU

3. **Training** (`src/dae/train.py`)
   - `train_dae()` implements training loop
   - Adam optimizer with configurable learning rate
   - **Key:** Loss computed ONLY on masked values (not entire reconstruction)
   - Adds Gaussian noise to inputs during training for denoising
   - Training input: corrupted data + noise → target: original masked values

4. **Evaluation** (`src/dae/evaluate.py`)
   - `evaluate_dae()` computes metrics on test predictions
   - Metrics (R², RMSE, MSE, MAE) computed ONLY on masked values
   - `aggregate_results()` combines metrics across multiple seeds
   - Saves predictions, targets, and metrics as separate files

5. **Visualization** (`src/dae/plots.py`)
   - `generate_all_dae_plots()` creates publication-ready figures
   - Loss curves, R² bar charts, predicted vs. truth scatter plots
   - Matches paper figures (Figures 3-7)

### KNN Pipeline Flow

1. **Imputation** (`src/knn/imputation.py`)
   - `KNNImputer` class wraps sklearn's NearestNeighbors
   - Configurable: n_neighbors, weights (uniform/distance), metric (euclidean/manhattan/cosine)
   - No training required - direct imputation based on nearest neighbors
   - Parallel execution using joblib (n_jobs=-1)

2. **Evaluation** (`src/knn/evaluate.py`)
   - Same metrics as DAE: R², RMSE, MSE, MAE
   - Tracks imputation time for performance comparison
   - `compare_knn_configs()` ranks configurations by performance

3. **Visualization** (`src/knn/plots.py`)
   - K-neighbor comparison plots
   - Weights/metric comparison visualizations

### Orchestration

**Entry Points:**
- `run_all.py` - Master orchestrator for all experiments
- `run_dae.py` - DAE experiments entry point
- `run_knn.py` - KNN experiments entry point
- `run_baseline.py` - Baseline experiments entry point
- `run_comparison.py` - Comparison generation entry point

**Configuration** (`src/config.py`):
- `DAEConfig` - DAE hyperparameters and paths
- `KNNConfig` - KNN parameters and paths
- `BaselineConfig` - Baseline parameters and paths
- `ComparisonConfig` - Comparison paths
- `get_config()` - Factory function for quick-test mode

**DAE Pipeline** (`src/dae/experiments.py`):
- `run_all_experiments()` uses `itertools.product()` to test all combinations
- `run_single_experiment()` runs one config across multiple seeds
- `generate_plots()` creates DAE visualizations
- Results saved to `results/dae/{models,metrics,plots}/`

**KNN Pipeline** (`src/knn/experiments.py`):
- `run_all_experiments()` supports parallel execution (12 experiments concurrently)
- `_run_experiment_wrapper()` enables joblib parallelization
- `generate_plots()` creates KNN visualizations
- Results saved to `results/knn/{metrics,predictions,plots}/`

**Baseline Pipeline** (`src/baselines/experiments.py`):
- `run_all_experiments()` runs zero imputation baseline
- `generate_plots()` creates baseline visualizations
- Results saved to `results/baselines/{metrics,predictions,plots}/`

**Comparison Pipeline** (`run_comparison.py`):
- `generate_all_comparisons()` creates cross-method comparisons
- Loads results from `results/{dae,knn,baselines}/summary.json`
- Generates bar charts, scatter plots, and text tables
- Results saved to `results/comparisons/`

## Critical Architecture Details

### Data Corruption Strategy
- **Two-step corruption:** (1) Mask random values to zero, (2) Add Gaussian noise to ALL values
- Mask is binary (True = was masked/corrupted, False = kept original)
- During training, noise is re-sampled every epoch for additional regularization

### Loss Computation
- **Masked MSE loss:** Only computes loss on values that were artificially corrupted
- Original zero values (unused ingredients) are NOT included in loss
- This focuses the model on imputing actually missing data, not sparse structure

### Device Support
Priority order: CUDA > MPS (Apple Silicon) > CPU
- Automatically detects and uses best available device
- MPS support requires macOS 12.3+ and PyTorch 2.0+

### Filename Convention

**DAE files** follow pattern: `miss{rate}_lr{lr}_n{neurons}_ep{epochs}_seed{seed}.{ext}`
- Example: `miss0.01_lr0.001_n512_ep1000_seed42.pt`

**KNN files** follow pattern: `miss{rate}_k{neighbors}_{weights}_{metric}_seed{seed}.{ext}`
- Example: `miss0.01_k5_distance_euclidean_seed42.npz`

## Key Experimental Variables

### DAE Configuration (from `ExperimentConfig` in main.py:23-43)
- **Missingness rates:** 1%, 5%, 10% (percentage of data to corrupt)
- **Learning rates:** 10⁻¹, 10⁻³, 10⁻⁵ (paper shows 10⁻³ is optimal)
- **Neuron sizes:** 256, 512, 1024 (both hidden and latent layers)
- **Epochs:** 100, 500, 1000, 1200
- **Seeds:** 42, 50, 100 (for statistical robustness)
- **Total:** 243 experiments (3×3×3×4 configs × 3 seeds)

### KNN Configuration (from `KNNConfig` in knn_main.py:29-54)
- **Missingness rates:** 1%, 5%, 10% (same as DAE)
- **N-neighbors:** 3, 5, 10, 20, 50
- **Weights:** uniform, distance
- **Metrics:** euclidean, manhattan, cosine
- **Seeds:** 42, 50, 100 (matches DAE for fair comparison)
- **Parallelization:** 12 concurrent experiments, all CPU cores for nearest neighbor search
- **Total:** 90 experiments (3×5×2×3 configs × 3 seeds)

## Results Structure

```
results/
├── dae/                          # DAE experiment outputs
│   ├── models/                   # Trained model checkpoints (.pt)
│   ├── metrics/                  # Metrics JSON + predictions NPZ
│   │   ├── *_loss.json           # Loss history per epoch
│   │   ├── *_metrics.json        # Aggregated R²/RMSE stats
│   │   └── *_predictions.npz     # Predictions + targets arrays
│   ├── plots/                    # DAE-specific figures
│   │   ├── loss_curves_*.png
│   │   ├── r2_bars_*.png
│   │   └── pred_vs_truth_*.png
│   └── summary.json              # All DAE experiment results
├── knn/                          # KNN experiment outputs
│   ├── metrics/                  # KNN metrics JSON
│   │   └── *_metrics.json        # R²/RMSE + imputation time
│   ├── predictions/              # KNN predictions NPZ
│   │   └── *_predictions.npz     # Predictions + targets arrays
│   ├── plots/                    # KNN-specific figures
│   │   ├── k_comparison_*.png
│   │   └── weights_comparison_*.png
│   └── summary.json              # All KNN experiment results
├── missforest/                   # MissForest experiment outputs
│   ├── metrics/                  # MissForest metrics JSON
│   │   └── *_metrics.json        # R²/RMSE + imputation time
│   ├── predictions/              # MissForest predictions NPZ
│   │   └── *_predictions.npz     # Predictions + targets arrays
│   ├── plots/                    # MissForest-specific figures
│   │   ├── missforest_r2_bars.png
│   │   └── miss*_predictions.png
│   └── summary.json              # All MissForest experiment results
├── baselines/                    # Baseline method outputs
│   ├── metrics/
│   ├── predictions/
│   ├── plots/
│   └── summary.json
└── comparisons/                  # Cross-method comparison outputs
    ├── method_comparison.png     # R² bars: DAE vs KNN vs MissForest vs Zero
    ├── performance_vs_time.png   # Performance vs. computation scatter
    └── comparison_table.txt      # Detailed comparison table
```

## Important Implementation Notes

### DAE Training Details
- Batch training NOT used - entire dataset processed at once (1570+ samples fits in memory)
- Noise is added during both training AND evaluation for consistency
- Model saved after training completes (not during training)
- Uses PyTorch tensors throughout the pipeline

### KNN Implementation Details
- No training phase - KNN is a lazy learning method
- Uses numpy arrays (converted from PyTorch tensors)
- Parallel execution at two levels:
  1. **Experiment-level:** 12 experiments run concurrently via joblib
  2. **Neighbor search:** All CPU cores used for NearestNeighbors (n_jobs=-1)
- Much faster than DAE (~15-30 min for all 90 experiments vs. ~6-8 hours for DAE)
- Imputation time tracked for performance comparison

### MissForest Implementation Details
- Uses **sklearn's IterativeImputer** with **RandomForestRegressor** (not PyPI MissForest package)
- **Why sklearn?** PyPI MissForest caused segmentation faults with 99% sparse pharmaceutical data
- **Computational complexity:** Trains one RandomForest per feature (336 features = 336 models per iteration)
- **Performance:** Slower than KNN, faster than DAE (~4-8 hours for full experiments)
- **Algorithm:** Iterative chained equations - each feature predicted by all other features
- Uses numpy arrays (converted from PyTorch tensors)
- NaN used to mark missing values (sklearn convention)
- **Default settings:** max_iter=10, n_estimators=100 per RandomForest
- **Quick-test settings:** max_iter=5, n_estimators=10 (~5-10 min)

### Data Sparsity Handling
- Dataset is ~99% sparse (zeros for unused ingredients)
- Normalization preserves zeros (MinMaxScaler fits on entire range including zeros)
- All methods (DAE, KNN, MissForest) must distinguish between "unused ingredient" (original zero) and "missing value" (masked non-zero)
- KNN handles sparsity naturally through distance metrics
- DAE learns sparsity pattern through training
- MissForest handles sparsity through RandomForest feature importance

### Reproducibility
- Fixed seeds (42, 50, 100) used for data corruption across ALL methods (DAE, KNN, MissForest, baselines)
- Seeds control: data corruption mask creation, noise generation, model initialization (DAE only)
- Same corrupted data used for fair comparison between methods
- Results aggregated across seeds with mean ± std reported

## Data Requirements

- **File:** `data/material_name_smilesRemoved.csv`
- **Structure:** First 6 columns are metadata (id, article, author, formulation_id, operator, reviewer), remaining columns are ingredient compositions
- **Values:** Composition percentages (0-100% w/w)

## Paper Findings (for context)

### DAE Performance
- **1% missing:** R²=0.94±0.03 (1024 neurons, 1200 epochs, LR=10⁻³)
- **5% missing:** R²=0.48±0.09 (256 neurons, lower epochs, LR=10⁻³)
- **10% missing:** R²=0.37±0.05 (256 neurons, lower epochs, LR=10⁻³)
- **Key insight:** Learning rate has strongest effect; 10⁻³ performs best across all missingness levels
- **Capacity tradeoff:** Larger models (1024) better for low missingness; smaller models (256) generalize better at high missingness

### Method Comparison
The codebase now supports comprehensive comparison between:
- **DAE:** Neural network approach with denoising (deep learning)
- **MissForest:** Iterative Random Forest imputation (ensemble learning)
- **KNN:** Classical ML baseline (no training, instance-based)
- **Zero imputation:** Naive baseline (fill with zeros)

Use `poetry run python run_comparison.py` to create side-by-side performance comparisons after running experiments.
