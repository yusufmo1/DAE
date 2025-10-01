# DAE for Pharmaceutical Formulation Data Imputation

This pipeline reproduces the results from the paper:
**"Machine Learning Recovers Corrupted Pharmaceutical 3D Printing Formulation Data"**
by Olima Uddin, Yusuf Ali Mohammed, Simon Gaisford, and Moe Elbadawi

## Overview

The pipeline implements **Denoising Autoencoders (DAEs)** to impute missing values in pharmaceutical formulation data. The dataset contains 1,570+ formulations across 382 ingredients with ~99% sparsity (zeros for unused ingredients).

### Key Features

- ✅ **MPS Support**: Automatically uses Apple Silicon GPU (MPS) for training on Mac
- ✅ **Reproducible**: Uses fixed random seeds (42, 50, 100) for statistical robustness
- ✅ **Comprehensive**: Tests 27 configurations per missingness level
- ✅ **Publication-Ready**: Generates plots matching the paper figures

## Installation

### 1. Clone/Navigate to the directory

```bash
cd ollima
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Quick Test (Recommended First)

Test the pipeline with reduced parameters (~5 minutes):

```bash
python main.py --quick-test
```

This runs 1 experiment with:
- 1% missingness
- Learning rate: 10⁻³
- 256 neurons
- 100 epochs
- 1 seed

### Run Full Experiments

Reproduce all paper results (~several hours):

```bash
python main.py
```

This runs **243 experiments**:
- 3 missingness rates (1%, 5%, 10%)
- 3 learning rates (10⁻¹, 10⁻³, 10⁻⁵)
- 3 neuron sizes (256, 512, 1024)
- 3 epoch settings (100, 500, 1000, 1200)
- 3 seeds per experiment

### Custom Experiments

Run specific configurations:

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

### Generate Plots Only

If you already ran experiments and just want to regenerate plots:

```bash
python main.py --skip-training
```

## Project Structure

```
ollima/
├── data/
│   └── material_name_smilesRemoved.csv    # Formulation dataset
├── src/
│   ├── data_preprocessing.py              # Data loading & corruption
│   ├── model.py                           # DAE architecture
│   ├── train.py                           # Training logic
│   ├── evaluate.py                        # Metrics computation
│   └── visualize.py                       # Plot generation
├── results/
│   ├── models/                            # Trained model checkpoints
│   ├── metrics/                           # JSON metrics & loss histories
│   ├── plots/                             # Generated figures
│   └── summary.json                       # Aggregated results
├── requirements.txt                       # Python dependencies
├── main.py                                # Main pipeline orchestrator
└── README.md                              # This file
```

## Output Files

### Trained Models

Saved to `results/models/`:
- Filename format: `miss{rate}_lr{lr}_n{neurons}_ep{epochs}_seed{seed}.pt`
- Example: `miss0.01_lr0.001_n512_ep1000_seed42.pt`

### Metrics

Saved to `results/metrics/`:
- Loss histories: `*_loss.json`
- Aggregated metrics: `*_metrics.json`
- Predictions: `*_predictions.npz`

### Plots

Saved to `results/plots/`:
- Loss curves (Figure 3): `loss_curves_{pct}pct.png`
- R² bar plots (Figures 4-6): `r2_bars_{pct}pct.png`
- Predicted vs Truth (Figure 7): `pred_vs_truth_1pct.png`

## Key Results

The paper reports these R² scores for imputation:

| Missing Data | Best R² (mean ± std) | Configuration |
|--------------|----------------------|---------------|
| 1%           | 0.94 ± 0.03         | 1024 neurons, 1200 epochs, LR=10⁻³ |
| 5%           | 0.48 ± 0.09         | 256 neurons, lower epochs, LR=10⁻³ |
| 10%          | 0.37 ± 0.05         | 256 neurons, lower epochs, LR=10⁻³ |

### Key Findings

1. **Learning rate has the strongest effect** on performance
   - 10⁻³ performs best across all missingness levels
   - 10⁻¹ and 10⁻⁵ produce negative R² scores (worse than baseline)

2. **Larger models work better for low missingness** (1%)
   - 1024 neurons optimal for 1% missing

3. **Smaller models work better for high missingness** (5-10%)
   - 256 neurons optimal for higher missingness
   - Suggests smaller models generalize better with less signal

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

Each module can be used independently:

### Data Preprocessing

```python
from src.data_preprocessing import FormulationDataPreprocessor

preprocessor = FormulationDataPreprocessor('data/material_name_smilesRemoved.csv')
preprocessor.load_data()
preprocessor.normalize_data()

# Corrupt 1% of data
original, corrupted, mask = preprocessor.prepare_data(missingness_rate=0.01, seed=42)
```

### Model Creation

```python
from src.model import create_dae, get_device

device = get_device()
model = create_dae(input_dim=382, neuron_size=512, device=device)
```

### Training

```python
from src.train import train_dae

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

### Evaluation

```python
from src.evaluate import evaluate_dae

predictions, metrics = evaluate_dae(
    model=trained_model,
    original_data=original,
    corrupted_data=corrupted,
    mask=mask,
    device=device
)

print(f"R²: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Visualization

```python
from src.visualize import create_all_plots

create_all_plots('results', missingness_rates=[0.01, 0.05, 0.10])
```

## Citation

If you use this code, please cite:

```
Uddin, O., Mohammed, Y. A., Gaisford, S., & Elbadawi, M. (2025).
Machine Learning Recovers Corrupted Pharmaceutical 3D Printing Formulation Data.
[Journal/Conference details to be added]
```

## License

[Add appropriate license]

## Contact

For questions or issues:
- Corresponding Author: m.elbadawi@qmul.ac.uk
- Repository: [Add GitHub URL if applicable]

## Troubleshooting

### MPS Not Available

If you get "MPS not available" on Mac:
- Update to macOS 12.3+ and PyTorch 2.0+
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory

If training crashes with OOM:
1. Reduce neuron sizes: `--neuron-sizes 256`
2. Train on CPU (slower): Edit `model.py` to force CPU

### Missing Data File

Ensure `data/material_name_smilesRemoved.csv` exists:
```bash
ls data/material_name_smilesRemoved.csv
```

## Acknowledgments

- Queen Mary University of London
- UCL School of Pharmacy
