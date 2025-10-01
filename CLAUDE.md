# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Denoising Autoencoders (DAEs)** for imputing missing values in pharmaceutical 3D printing formulation data. The dataset contains 1,570+ formulations across 382 ingredients with ~99% sparsity (most ingredients are unused in each formulation).

Paper: "Machine Learning Recovers Corrupted Pharmaceutical 3D Printing Formulation Data" by Olima Uddin, Yusuf Ali Mohammed, Simon Gaisford, and Moe Elbadawi.

## Key Commands

### Running Experiments

```bash
# Quick test (5 min, recommended first run)
python main.py --quick-test

# Full experiments (~6-8 hours on Apple Silicon, ~24-48h on CPU)
python main.py

# Custom configurations
python main.py --missingness 0.01 --learning-rates 0.001 --neuron-sizes 512 --epochs 1000
python main.py --seeds 1 2 3 4 5

# Generate plots only (skip training)
python main.py --skip-training
```

### Running Individual Modules

Each module in `src/` can be run independently for testing:
```bash
python src/data_preprocessing.py
python src/model.py
python src/train.py
python src/evaluate.py
python src/visualize.py
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## Architecture & Data Flow

### Pipeline Flow

1. **Data Preprocessing** (`src/data_preprocessing.py`)
   - Loads CSV, extracts ingredient columns (skips 6 metadata columns)
   - Normalizes to [0, 1] using MinMaxScaler
   - Creates corruption: randomly masks values → sets to zero → adds Gaussian noise (σ=0.1)

2. **Model Creation** (`src/model.py`)
   - DAE architecture: `Input → [FC+BN+LeakyReLU] → [FC+BN+LeakyReLU] → [FC+Sigmoid] → Output`
   - Both hidden and latent layers use same neuron count (256/512/1024)
   - Overcomplete architecture (latent_dim = hidden_dim = neuron_size)

3. **Training** (`src/train.py`)
   - Adam optimizer with configurable learning rate
   - **Key:** Loss computed ONLY on masked values (not entire reconstruction)
   - Adds Gaussian noise to inputs during training for denoising
   - Training input: corrupted data + noise → target: original masked values

4. **Evaluation** (`src/evaluate.py`)
   - Metrics (R², RMSE, MSE, MAE) computed ONLY on masked values
   - Saves predictions, targets, and metrics as separate files

5. **Visualization** (`src/visualize.py`)
   - Loss curves, R² bar charts, predicted vs. truth scatter plots
   - Matches paper figures (Figures 3-7)

### Orchestration

`main.py` orchestrates the full pipeline:
- `ExperimentConfig` class centralizes all hyperparameters
- `run_single_experiment()` runs one config across multiple seeds
- `run_all_experiments()` uses `itertools.product()` to test all combinations
- Results saved to `results/models/`, `results/metrics/`, `results/plots/`

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
Models/metrics follow pattern: `miss{rate}_lr{lr}_n{neurons}_ep{epochs}_seed{seed}.{ext}`
- Example: `miss0.01_lr0.001_n512_ep1000_seed42.pt`

## Key Experimental Variables

From `ExperimentConfig` in main.py:76-107:
- **Missingness rates:** 1%, 5%, 10% (what percentage of data to corrupt)
- **Learning rates:** 10⁻¹, 10⁻³, 10⁻⁵ (paper shows 10⁻³ is optimal)
- **Neuron sizes:** 256, 512, 1024 (both hidden and latent layers)
- **Epochs:** 100, 500, 1000, 1200
- **Seeds:** 42, 50, 100 (for statistical robustness)
- **Total:** 243 experiments (3×3×3×3 configs × 3 seeds)

## Results Structure

```
results/
├── models/           # Trained model checkpoints (.pt)
├── metrics/          # Metrics JSON + predictions NPZ
│   ├── *_loss.json        # Loss history per epoch
│   ├── *_metrics.json     # Aggregated R²/RMSE stats
│   └── *_predictions.npz  # Predictions + targets arrays
├── plots/            # Publication-ready figures
│   ├── loss_curves_*.png
│   ├── r2_bars_*.png
│   └── pred_vs_truth_*.png
└── summary.json      # All experiment results
```

## Important Implementation Notes

### Training Details
- Batch training NOT used - entire dataset processed at once (1570+ samples fits in memory)
- Noise is added during both training AND evaluation for consistency
- Model saved after training completes (not during training)

### Data Sparsity Handling
- Dataset is ~99% sparse (zeros for unused ingredients)
- Normalization preserves zeros (MinMaxScaler fits on entire range including zeros)
- Model must learn to distinguish between "unused ingredient" (original zero) and "missing value" (masked non-zero)

### Reproducibility
- Fixed seeds (42, 50, 100) used for data splitting and corruption
- Seeds control: data corruption mask creation, noise generation, model initialization
- Results aggregated across seeds with mean ± std reported

## Data Requirements

- **File:** `data/material_name_smilesRemoved.csv`
- **Structure:** First 6 columns are metadata (id, article, author, formulation_id, operator, reviewer), remaining columns are ingredient compositions
- **Values:** Composition percentages (0-100% w/w)

## Paper Findings (for context)

- **1% missing:** R²=0.94±0.03 (1024 neurons, 1200 epochs, LR=10⁻³)
- **5% missing:** R²=0.48±0.09 (256 neurons, lower epochs, LR=10⁻³)
- **10% missing:** R²=0.37±0.05 (256 neurons, lower epochs, LR=10⁻³)
- **Key insight:** Learning rate has strongest effect; 10⁻³ performs best across all missingness levels
- **Capacity tradeoff:** Larger models (1024) better for low missingness; smaller models (256) generalize better at high missingness
