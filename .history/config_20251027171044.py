"""
Runtime configuration for condition recognition strategy.
Switch between 'EVM' and 'HMM' without changing code.
"""

# One of: 'EVM', 'HMM'
RECOGNITION_STRATEGY = 'HMM'

# EVM params
EVM_TAIL_FRAC = 0.5
EVM_REJECT_THRESHOLD = 0.5

# HMM params
HMM_N_STATES = 4
HMM_COV_REG = 1e-6
HMM_N_ITER = 25
HMM_GLRT_QUANTILE = 0.15  # lower quantile -> stricter open-set rejection

# HMM feature options
HMM_FEATURE_MODE = 'stft_bands'  # 'stft' | 'stft_bands' | 'raw' | 'raw_win'
HMM_NUM_BANDS = 6
HMM_RAW_WIN = 64
HMM_RAW_HOP = 32
HMM_STANDARDIZE = True
HMM_RESTARTS = 3
HMM_SELF_TRANSITION = 0.93
HMM_USE_AVG_LL = True

# Reproducibility
RANDOM_STATE = 0


