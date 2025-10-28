"""
Runtime configuration for condition recognition strategy.
Switch between 'EVM', 'HMM', and 'DTW' without changing code.
"""

# One of: 'EVM', 'HMM', 'DTW'
RECOGNITION_STRATEGY = 'HMM'

# EVM params
EVM_TAIL_FRAC = 0.5
EVM_REJECT_THRESHOLD = 0.5

# HMM params
HMM_N_STATES = 3
HMM_COV_REG = 1e-6
HMM_N_ITER = 15
HMM_GLRT_QUANTILE = 0.1  # lower quantile -> stricter open-set rejection

# DTW params
# Window band as a fraction of the min(lengths); None disables banding (slower)
DTW_WINDOW_RATIO = 0.1
# Downsample raw time series by this stride before DTW (>=1)
DTW_DOWNSAMPLE = 4
# Quantile of in-class nearest DTW distances to set open-set threshold (higher -> stricter Unknown)
DTW_REJECT_QUANTILE = 0.9
# Safety margin multiplier on threshold (>=1.0)
DTW_MARGIN = 1.2

# Reproducibility
RANDOM_STATE = 0


