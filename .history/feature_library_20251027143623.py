import os
import glob
import numpy as np
import scipy.io as scio
from typing import List, Dict
from STFT import compute_stft, read_mat


def extract_condition_prefix(names: List[str]) -> str:
    return names[0].split('_')[0]


def build_features_for_condition(train_names: List[str]) -> np.ndarray:
    """
    Build STFT-based feature rows for each dataset under a condition group.
    Each row corresponds to one dataset in train_names.
    We re-use compute_stft + read_mat to build a compact feature vector per dataset.
    """
    feats = []
    for name in train_names:
        # Re-use compute_stft to get PCA-reduced signal x and target y
        fs, f, t, z, x, y = compute_stft(name)
        # basic STFT features: statistics on magnitude
        Z = np.abs(z)
        # summarize per-time and per-frequency
        f_mean = np.mean(Z, axis=0)
        f_std = np.std(Z, axis=0)
        t_mean = np.mean(Z, axis=1)
        t_std = np.std(Z, axis=1)
        # select a compact set: top-k from each summary to keep dimension small
        kf = min(16, f_mean.shape[0])
        kt = min(16, t_mean.shape[0])
        feat = np.hstack([
            np.sort(f_mean)[-kf:], np.sort(f_std)[-kf:],
            np.sort(t_mean)[-kt:], np.sort(t_std)[-kt:],
            [np.mean(x), np.std(x)],
        ])
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


