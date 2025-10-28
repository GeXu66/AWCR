"""
Runtime configuration for condition recognition strategy and features.
"""

# One of: 'EVM', 'HMM'
RECOGNITION_STRATEGY = 'EVM'

# EVM params
EVM_TAIL_FRAC = 0.5
EVM_REJECT_THRESHOLD = 0.5

# HMM params
HMM_N_STATES = 3
HMM_COV_REG = 1e-6
HMM_N_ITER = 15
HMM_GLRT_QUANTILE = 0.1  # lower quantile -> stricter open-set rejection

# Reproducibility
RANDOM_STATE = 0


# Feature backend: 'STFT' (legacy) or 'WST' (Wavelet scattering substitute + MSC/CCA)
FEATURE_BACKEND = 'WST'

# Feature alignment across batches: 'NONE', 'CORAL', or 'WHITEN'
FEATURE_ALIGN = 'CORAL'

# For CORAL: which batch suffix is the reference (e.g., '01')
REFERENCE_BATCH_SUFFIX = '01'

# Alignment numerical stability epsilon
ALIGN_EPS = 1e-6

# WST options (effective if FEATURE_BACKEND == 'WST')
WST_ORDER = 2

# Physical bands for spectral features (Hz). Adjust to vehicle dynamics.
PHYSICAL_BANDS = [
    (0.5, 5.0),    # steering and low-frequency dynamics
    (5.0, 15.0),   # chassis/body
    (15.0, 40.0),  # wheel/suspension
]


