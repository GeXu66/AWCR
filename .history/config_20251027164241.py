"""
Runtime configuration for condition recognition strategy.
Switch between 'EVM' and 'HMM' without changing code.
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


