import os
import glob
import numpy as np
import scipy.io as scio
from typing import List, Dict
from STFT import compute_stft, read_mat
from sklearn.decomposition import NMF
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
import config


def extract_condition_prefix(names: List[str]) -> str:
    token = names[0].split('_')[0]
    # Use coarse class by stripping digits, e.g., 'SW20' -> 'SW'
    coarse = ''.join([ch for ch in token if ch.isalpha()])
    return coarse


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

    # Base 10-dim features from |STFT| and time signal
    base_feat = _fixed_length_feature_from_stft(Z, x)

    # Build NMF-based features per component (15 features each)
    #  - Use original multi-channel X for time-domain statistics
    trainFile = ['../Data/matdata/' + name + '.mat']
    X, _ = read_mat(trainFile)

    nmf_n = int(getattr(config, 'NMF_N_COMPONENTS', 1))
    nmf_random_state = int(getattr(config, 'RANDOM_STATE', 0))
    nmf_init = getattr(config, 'NMF_INIT', 'nndsvda')
    nmf_max_iter = int(getattr(config, 'NMF_MAX_ITER', 1000))
    nmf_tol = float(getattr(config, 'NMF_TOL', 1e-4))

    # Time-domain statistics across channels (median per-channel kurtosis/skewness)
    # Replicated per component to form 15 features/component
    kurt_list = []
    skew_list = []
    for col in X.T:
        v = np.squeeze(col)
        try:
            k = float(sp_kurtosis(v, fisher=True, bias=False))
        except Exception:
            k = float('nan')
        try:
            s = float(sp_skew(v, bias=False))
        except Exception:
            s = float('nan')
        kurt_list.append(k)
        skew_list.append(s)
    kurt_time = float(np.nanmedian(np.array(kurt_list)))
    skewness_time = float(np.nanmedian(np.array(skew_list)))

    # NMF decomposition on |Z|: Z ~ W @ H, W:(F,K), H:(K,T)
    F, T = Z.shape
    nmf_n = max(1, min(int(nmf_n), min(F, T)))
    nmf_model = NMF(
        n_components=nmf_n,
        init=nmf_init,
        random_state=nmf_random_state,
        max_iter=nmf_max_iter,
        tol=nmf_tol,
    )
    W = nmf_model.fit_transform(Z)
    H = nmf_model.components_

    # Per-component 15-dim features
    comp_feats = []
    alpha = 2.0
    # Renyi entropy computed on the full normalized |Z| to mirror compute_tf_feature
    sZ = float(np.sum(Z))
    if sZ > 0.0:
        pZ = Z / sZ
        renyi_entropy_global = float((1.0 / (1.0 - alpha)) * np.log2(np.sum(np.power(pZ, alpha))))
    else:
        renyi_entropy_global = 0.0
    for k in range(nmf_n):
        w = np.maximum(W[:, k], 0.0)
        h = np.maximum(H[k, :], 0.0)

        rows = w.size
        cols = h.size

        # Sparsity (following the style used in STFT.compute_tf_feature)
        l1_w = float(np.sum(w))
        l2sq_w = float(np.sum(w * w)) + 1e-12
        sparsity_coff_vector = (np.sqrt(rows) - l1_w / l2sq_w) / (np.sqrt(rows) - 1.0 + 1e-12)

        l1_h = float(np.sum(h))
        l2sq_h = float(np.sum(h * h)) + 1e-12
        sparsity_base_vector = (np.sqrt(cols) - l1_h / l2sq_h) / (np.sqrt(cols) - 1.0 + 1e-12)

        # Smoothness (discontinuity)
        discontinuity_coff_vector = float(np.linalg.norm(np.diff(w), 2)) if rows > 1 else 0.0
        discontinuity_base_vector = float(np.linalg.norm(np.diff(h), 2)) if cols > 1 else 0.0

        # Use global Renyi entropy same as compute_tf_feature
        renyi_entropy = renyi_entropy_global

        # Statistics
        max_coff_vector = float(np.max(w)) if rows > 0 else 0.0
        max_base_vector = float(np.max(h)) if cols > 0 else 0.0
        std_coff_vector = float(np.std(w, ddof=1)) if rows > 1 else 0.0
        std_base_vector = float(np.std(h, ddof=1)) if cols > 1 else 0.0

        # Higher-order moments
        try:
            kurt_coff_vector = float(sp_kurtosis(w, fisher=True, bias=False)) if rows > 3 else 0.0
        except Exception:
            kurt_coff_vector = 0.0
        try:
            kurt_base_vector = float(sp_kurtosis(h, fisher=True, bias=False)) if cols > 3 else 0.0
        except Exception:
            kurt_base_vector = 0.0
        try:
            skewness_coff_vector = float(sp_skew(w, bias=False)) if rows > 2 else 0.0
        except Exception:
            skewness_coff_vector = 0.0
        try:
            skewness_base_vector = float(sp_skew(h, bias=False)) if cols > 2 else 0.0
        except Exception:
            skewness_base_vector = 0.0

        comp_feat = [
            sparsity_coff_vector,
            sparsity_base_vector,
            discontinuity_coff_vector,
            discontinuity_base_vector,
            renyi_entropy,
            max_coff_vector,
            max_base_vector,
            std_coff_vector,
            std_base_vector,
            kurt_coff_vector,
            kurt_base_vector,
            skewness_coff_vector,
            skewness_base_vector,
            kurt_time,
            skewness_time,
        ]
        comp_feats.append(np.array(comp_feat, dtype=np.float32))

    if len(comp_feats) > 0:
        nmf_feat = np.concatenate(comp_feats, axis=0)
    else:
        nmf_feat = np.zeros((0,), dtype=np.float32)

    full_feat = np.concatenate([base_feat.astype(np.float32, copy=False), nmf_feat], axis=0)
    return full_feat


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


