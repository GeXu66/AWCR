import os
import glob
import numpy as np
import scipy.io as scio
from typing import List, Dict
from STFT import compute_stft, read_mat


def extract_condition_prefix(names: List[str]) -> str:
    return names[0].split('_')[0]


def _fixed_length_feature_from_stft(abs_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Create a fixed-length feature vector from |STFT| matrix and 1D signal x.
    Features (10 dims):
    [mean(|Z|), std(|Z|), qf10, qf50, qf90, qt10, qt50, qt90, mean(x), std(x)]
    where qf* are quantiles over frequency-aggregated means, qt* over time-aggregated means.
    """
    Z = abs_z
    meanZ = float(np.mean(Z))
    stdZ = float(np.std(Z))
    # frequency aggregation: mean over time -> vector length n_freq
    f_profile = np.mean(Z, axis=1)
    # time aggregation: mean over freq -> vector length n_time
    t_profile = np.mean(Z, axis=0)
    qf10, qf50, qf90 = np.percentile(f_profile, [10, 50, 90])
    qt10, qt50, qt90 = np.percentile(t_profile, [10, 50, 90])
    meanx = float(np.mean(x))
    stdx = float(np.std(x))
    feat = np.array([meanZ, stdZ, qf10, qf50, qf90, qt10, qt50, qt90, meanx, stdx], dtype=np.float32)
    return feat


def compute_feature_for_name(name: str) -> np.ndarray:
    fs, f, t, z, x, y = compute_stft(name)
    Z = np.abs(z)
    x = np.squeeze(x)
    return _fixed_length_feature_from_stft(Z, x)


def build_features_for_condition(train_names: List[str]) -> np.ndarray:
    """
    Build STFT-based feature rows for each dataset under a condition group.
    Each row corresponds to one dataset in train_names.
    We re-use compute_stft + read_mat to build a compact feature vector per dataset.
    """
    feats = []
    for name in train_names:
        feat = compute_feature_for_name(name)
        feats.append(feat.reshape(1, -1))
    F = np.vstack(feats)
    return F


def build_feature_library(trainlist: List[List[str]], feature_dir: str = 'feature') -> Dict[str, str]:
    os.makedirs(feature_dir, exist_ok=True)
    saved = {}
    for group in trainlist:
        prefix = extract_condition_prefix(group)
        F = build_features_for_condition(group)
        out_path = os.path.join(feature_dir, f'{prefix}.npy')
        np.save(out_path, F)
        saved[prefix] = out_path
        print(f'Feature saved: {prefix} -> {out_path}, shape {F.shape}')
    return saved


if __name__ == '__main__':
    trainlist = [
        ['SW20_01', 'SW20_02'], ['SW25_01', 'SW25_02'], ['SW35_01', 'SW35_02'],
        ['RE20_01', 'RE20_02'], ['RE30_01', 'RE30_02'], ['RE40_01', 'RE40_02'],
        ['CJ30_01', 'CJ30_02'], ['CJ40_01', 'CJ40_02'], ['CJ50_01', 'CJ50_02'],
    ]
    build_feature_library(trainlist)


