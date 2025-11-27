# Adaptive Working Condition Recognition (AWCR)

A framework for adaptive working condition recognition and multi-input multi-output (MIMO) system identification with open-set classification capabilities.

## Overview

This project implements an adaptive framework for recognizing and identifying unknown working conditions in dynamic systems. It combines time-frequency feature extraction, open-set classification, and noncausal MIMO FIR modeling to handle both known and unknown operating conditions with online model adaptation.

## Key Features

- **Open-Set Condition Recognition**: Supports two recognition strategies:
  - **EVM (Extreme Value Machine)**: Tail distribution-based open-set recognition using Weibull/GEV models
  - **HMM-GLRT**: Hidden Markov Model with Generalized Likelihood Ratio Test for sequential data

- **MIMO System Identification**: Noncausal FIR modeling with ridge regression for multi-channel input-output estimation

- **Adaptive Model Management**: Dynamic model creation and online adaptation for newly encountered conditions

- **Rich Feature Extraction**:
  - Short-Time Fourier Transform (STFT) based time-frequency analysis
  - Non-negative Matrix Factorization (NMF) for spectral decomposition
  - Statistical features: kurtosis, skewness, Rényi entropy, sparsity, discontinuity

## Project Structure

### Core Modules

- **[condition_pipeline.py](condition_pipeline.py)**: Main pipeline orchestrating condition recognition, model management, and evaluation
  - `pipeline()`: Sequential processing of test conditions with adaptive model update
  - `pipeline_exp3()`: Long-sequence segmentation and recognition

- **[MIMOFIR.py](MIMOFIR.py)**: MIMO FIR system identification
  - Noncausal FIR design matrix construction (past and future orders)
  - Ridge regression with automatic regularization
  - Training and prediction for multi-output systems

- **[recognizers.py](recognizers.py)**: Condition recognition strategies
  - `EVMRecognizer`: Open-set classifier using extreme value theory
  - `HMMGLRTRecognizer`: Sequential pattern recognition with GLRT-based rejection
  - `GaussianHMM`: Custom Gaussian HMM with Baum-Welch training

- **[feature_library.py](feature_library.py)**: Feature extraction and management
  - STFT-based time-frequency feature extraction
  - NMF decomposition with configurable components
  - Per-condition feature library construction

- **[evm_classifier.py](evm_classifier.py)**: Extreme Value Machine implementation
  - Tail distribution fitting on per-class distances
  - Survival probability-based scoring
  - Open-set rejection mechanism

- **[models_manager.py](models_manager.py)**: MIMO model persistence and lifecycle management
  - Model saving/loading with parameter tracking
  - Online model update via convex combination
  - Centralized model storage

### Auxiliary Modules

- **[STFT.py](STFT.py)**: Time-frequency analysis utilities
  - PCA-based dimensionality reduction
  - STFT computation with configurable parameters
  - Visualization tools for spectrograms

- **[config.py](config.py)**: Runtime configuration
  - Recognition strategy selection (EVM/HMM)
  - Hyperparameter tuning for NMF, EVM, HMM
  - Random state for reproducibility

- **[DataHeader.py](DataHeader.py)**: Working condition metadata and enumerations

- **[utils.py](utils.py)**: Utility functions for visualization and analysis

## Methodology

### 1. Feature Extraction

Each working condition is characterized by a fixed-length feature vector derived from:

1. **Base Features (10-dim)**:
   - Mean and std of STFT magnitude
   - Quantiles (10th, 50th, 90th) of frequency and time profiles
   - Time-domain statistics (mean, std)

2. **NMF-based Features (15-dim per component)**:
   - Sparsity coefficients for basis and activation vectors
   - Discontinuity measures (L2 norm of differences)
   - Rényi entropy of the full spectrogram
   - Max values and standard deviations
   - Higher-order statistics (kurtosis, skewness)
   - Time-domain kurtosis and skewness across channels

**Total dimension**: 10 + 15 × N_components (default: 10 + 75 = 85 dims)

### 2. Open-Set Recognition

#### EVM Strategy
- Compute class-specific means from training features
- Fit GEV distributions to tail distances (largest k%)
- Score test sample by survival probability: P(d > threshold | class)
- Reject as "Unknown" if max probability < threshold

#### HMM-GLRT Strategy
- Train per-class Gaussian HMMs on sequential features
- Train background model on all data
- Compute log-likelihood ratio: LLR = log P(X|class) - log P(X|background)
- Reject if LLR < calibrated threshold (quantile-based)

### 3. MIMO FIR Modeling

For a system with `ni` inputs and `no` outputs, the noncausal FIR model is:

```
y(t) = £ ˜_Ä · u(t-Ä)  for Ä = -future_order, ..., 0, ..., past_order
```

- **Design Matrix**: Constructed with edge padding (replicate first/last samples)
- **Parameter Estimation**: Ridge regression with condition number-based » selection
- **Model Update**: Convex combination of old and new parameters: `˜_new = (1-±)˜_old + ±˜_data`

### 4. Adaptive Pipeline

For each test condition:
1. **Recognition**: Predict working condition using EVM/HMM
2. **Known Condition**: Load existing model, estimate outputs, optionally update model
3. **Unknown Condition**:
   - Use fallback model (most similar known class) for estimation
   - Create new model from test data
   - Add to recognizer's unknown pool (max 3 unknown classes)
4. **Online Adaptation**: Incrementally update models with new observations (if enabled)

## Configuration

Key parameters in [config.py](config.py):

```python
# Recognition strategy: 'EVM' or 'HMM'
RECOGNITION_STRATEGY = 'EVM'

# EVM parameters
EVM_TAIL_FRAC = 0.5           # Fraction of distances for tail fitting
EVM_REJECT_THRESHOLD = 0.5    # Minimum probability for known class

# NMF parameters
NMF_N_COMPONENTS = 5          # Number of spectral components
NMF_MAX_ITER = 5000
NMF_INIT = 'nndsvda'          # Initialization method

# HMM parameters
HMM_N_STATES = 3              # Number of hidden states
HMM_COV_REG = 1e-5            # Covariance regularization
HMM_N_ITER = 30               # Baum-Welch iterations
HMM_GLRT_QUANTILE = 0.1       # GLRT threshold quantile

# General
RANDOM_STATE = 0
```

## Usage

### Basic Pipeline

```python
from condition_pipeline import pipeline

# Define training groups (each group = one condition type)
trainlist = [
    ['SW20_01', 'SW20_02'],  # Sway condition at 20% intensity
    ['RE20_01', 'RE20_02'],  # Regular condition at 20%
    ['CJ30_01', 'CJ30_02'],  # Cornering condition at 30%
]

# Define test samples (one per list for sequential processing)
testlist = [
    ['SW20_03'], ['RE20_03'], ['CJ30_03'], ['RC30_01']  # RC30 is unknown
]

# Channel configuration
channel_in = [69, 78, 83, 84, 91, 94, 95, 96]   # Input channels
channel_out = [32, 33, 34, 35, 36, 37]          # Output channels

# Run pipeline with model update enabled
pipeline(
    trainlist=trainlist,
    testlist=testlist,
    channel_in=channel_in,
    channel_out=channel_out,
    past_order=50,
    future_order=50,
    mean_flag=True,
    update_model=True,
    results_dir='result/EXP1'
)
```

### Long-Sequence Segmentation (EXP3)

```python
from condition_pipeline import pipeline_exp3

# Single long recording with multiple conditions
longlist = ['SW20_03', 'SW25_03', 'SW35_03', 'RE40_01', 'RE40_02']

pipeline_exp3(
    trainlist=trainlist,
    longlist=longlist,
    channel_in=channel_in,
    channel_out=channel_out,
    past_order=50,
    future_order=50,
    mean_flag=True,
    results_dir='result/EXP3'
)
```

### Direct MIMO FIR Training

```python
from MIMOFIR import fit_per_condition, evaluate_on_tests

# Train models per condition
thetas = fit_per_condition(
    past_order=50,
    future_order=50,
    channel_in=channel_in,
    channel_out=channel_out,
    mean_flag=True,
    train_groups=trainlist,
    piece='all'
)

# Evaluate on test data
results = evaluate_on_tests(
    thetas=thetas,
    past_order=50,
    future_order=50,
    channel_in=channel_in,
    channel_out=channel_out,
    mean_flag=True,
    train_groups=trainlist,
    test_groups=testlist,
    piece='all'
)
```

## Experiments

The project includes multiple experimental setups:

- **EXP1**: Basic known/unknown condition recognition with limited training data
- **EXP2**: Comprehensive evaluation across all condition types with full coverage
- **EXP3**: Long-sequence segmentation and online recognition

Results are saved with:
- Prediction MAT files (`Y_true`, `Y_pred`, `class`)
- Performance metrics (RMSE, R², MAPE)
- IEEE-style plots (PNG/PDF) for publications

## Data Format

Expected data structure:
```
../Data/matdata/
   SW20_01.mat
   SW20_02.mat
   RE20_01.mat
   ...
```

Each `.mat` file should contain:
- Multi-channel time-series data
- Key matching the filename (e.g., 'SW20_01')
- Columns: [various sensor channels]
  - Input channels: Specified in `channel_in`
  - Output channels: Specified in `channel_out`

## Output Structure

```
result/
   EXP1/
      SW20_01/
         update/prediction.mat
         update/metrics.txt
         update/SW20_01_ieee.png
         update/SW20_01_ieee.pdf
      combined_ieee.png
   EXP3/
      combined.mat
   MIMOFIR/
       SW20/
          SW20_train_pred.mat
          SW20_metrics.json
          SW20_train_ieee.pdf
       ...
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- (Optional) SciencePlots for IEEE-style figures

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib
# Optional for publication-quality plots
pip install SciencePlots
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{awcr2025,
  title={Adaptive Working Condition Recognition Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/AWCR}
}
```

## License

[Specify your license here]

## Contact

For questions or collaboration, please contact: [your.email@institution.edu]
