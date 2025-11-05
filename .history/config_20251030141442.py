"""
Runtime configuration for condition recognition strategy.
Switch between 'EVM' and 'HMM' without changing code.
"""

# One of: 'EVM', 'HMM'
RECOGNITION_STRATEGY = 'EVM'

# EVM params
EVM_TAIL_FRAC = 0.5
EVM_REJECT_THRESHOLD = 0.5

# NMF params
# Number of NMF components used when extracting per-condition features
NMF_N_COMPONENTS = 2
NMF_MAX_ITER = 5000
# 'nndsvda' gives a good non-random start and converges faster than 'random'
NMF_INIT = 'nndsvda'
NMF_TOL = 1e-4

# HMM params
HMM_N_STATES = 3
HMM_COV_REG = 1e-5
HMM_N_ITER = 30
HMM_GLRT_QUANTILE = 0.1  # lower quantile -> stricter open-set rejection

# Reproducibility
RANDOM_STATE = 0


