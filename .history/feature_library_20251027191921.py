import os
import glob
import numpy as np
import scipy.io as scio
from typing import List, Dict, Tuple
from STFT import compute_stft, read_mat
from scipy.signal import welch, coherence
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import config


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


def _load_multichannel_for_name(name: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load raw multi-channel signal matrix X (T x C) and target y for a given dataset name.
    Returns (X, y, fs)
    """
    trainFile = ['../Data/matdata/' + name + '.mat']
    X, y = read_mat(trainFile)
    fs = 512
    return X.astype(np.float64), np.squeeze(y).astype(np.float64), fs


def _compute_wst_features_per_channel(x: np.ndarray, fs: int, bands: List[Tuple[float, float]]) -> np.ndarray:
    """
    Wavelet scattering-like features per 1D channel.
    Try to use kymatio Scattering1D; if unavailable, fall back to CWT-based band energies.
    Output is a compact vector for this channel.
    """
    # Try kymatio first
    try:
        # Lazy import to avoid hard dependency if not installed
        from kymatio.numpy import Scattering1D  # type: ignore
        T = int(len(x))
        # choose J conservatively to limit feature size
        J = max(3, min(6, int(np.floor(np.log2(max(T, 8))) - 3)))
        Q = 8
        scattering = Scattering1D(J=J, shape=T, Q=Q, max_order=min(2, getattr(config, 'WST_ORDER', 2)))
        Sx = scattering(x)
        # Global average over time for invariance
        feat = np.mean(np.abs(Sx), axis=-1)
        # Limit dimension by taking first N coefficients if extremely long
        if feat.ndim > 1:
            feat = feat.reshape(-1)
        if feat.shape[0] > 128:
            feat = feat[:128]
        return feat.astype(np.float64)
    except Exception:
        # Fall back: CWT scalogram stats at band centers
        try:
            import pywt  # type: ignore
            wavelet = pywt.ContinuousWavelet('morl')
            dt = 1.0 / float(fs)
            centers = [0.5 * (lo + hi) for (lo, hi) in bands]
            # morlet central frequency
            fc = pywt.central_frequency(wavelet)
            # scale = fc / (f * dt)
            scales = [fc / (max(1e-6, f) * dt) for f in centers]
            coefs, _ = pywt.cwt(x, scales, wavelet, sampling_period=dt)
            A = np.abs(coefs)
            # For each scale: mean and std of amplitude over time
            mean_amp = np.mean(A, axis=1)
            std_amp = np.std(A, axis=1)
            return np.concatenate([mean_amp, std_amp]).astype(np.float64)
        except Exception:
            # Ultimate fallback: simple bandpower via Welch in the provided bands
            f, Pxx = welch(x, fs=fs, nperseg=min(1024, max(256, len(x) // 4)))
            total = np.trapz(Pxx, f) + 1e-12
            feats = []
            for lo, hi in bands:
                mask = (f >= lo) & (f <= hi)
                bp = np.trapz(Pxx[mask], f[mask])
                feats.append(bp / total)
            return np.asarray(feats, dtype=np.float64)


def _compute_multichannel_relationship_features(X: np.ndarray, fs: int, bands: List[Tuple[float, float]]) -> np.ndarray:
    """
    Multi-channel features: mean coherence per band across selected pairs, 
    band energy ratios (averaged over channels), and CCA top correlations.
    """
    T, C = X.shape
    # Band energies per channel
    nperseg = min(1024, max(256, T // 4))
    band_energy_ratios = []
    for c in range(C):
        f, Pxx = welch(X[:, c], fs=fs, nperseg=nperseg)
        total = np.trapz(Pxx, f) + 1e-12
        ratios = []
        for lo, hi in bands:
            mask = (f >= lo) & (f <= hi)
            bp = np.trapz(Pxx[mask], f[mask])
            ratios.append(bp / total)
        band_energy_ratios.append(ratios)
    band_energy_ratios = np.asarray(band_energy_ratios, dtype=np.float64)
    mean_band_energy = np.mean(band_energy_ratios, axis=0)  # per band

    # Coherence across simple adjacent pairs to bound cost
    max_pairs = min(8, C - 1)
    coh_per_band = []
    for i in range(max_pairs):
        f, Cxy = coherence(X[:, i], X[:, i + 1], fs=fs, nperseg=nperseg)
        for bi, (lo, hi) in enumerate(bands):
            mask = (f >= lo) & (f <= hi)
            val = float(np.mean(Cxy[mask])) if np.any(mask) else 0.0
            coh_per_band.append(val)
    # Average coherence per band over the evaluated pairs
    if len(coh_per_band) > 0:
        coh_per_band = np.asarray(coh_per_band, dtype=np.float64).reshape(max_pairs, len(bands))
        mean_coh = np.mean(coh_per_band, axis=0)
    else:
        mean_coh = np.zeros(len(bands), dtype=np.float64)

    # CCA between two channel groups (first half vs second half)
    split = C // 2
    if split >= 2 and (C - split) >= 2:
        X1 = X[:, :split]
        X2 = X[:, split:]
        scaler1 = StandardScaler(with_mean=True, with_std=True)
        scaler2 = StandardScaler(with_mean=True, with_std=True)
        X1z = scaler1.fit_transform(X1)
        X2z = scaler2.fit_transform(X2)
        n_comp = min(2, split, C - split)
        cca = CCA(n_components=n_comp, max_iter=500)
        try:
            U, V = cca.fit_transform(X1z, X2z)
            cca_corrs = []
            for k in range(n_comp):
                u = U[:, k]
                v = V[:, k]
                c = np.corrcoef(u, v)[0, 1]
                cca_corrs.append(float(np.clip(c, -1.0, 1.0)))
            cca_feats = np.asarray(cca_corrs, dtype=np.float64)
        except Exception:
            cca_feats = np.zeros(min(2, split, C - split), dtype=np.float64)
    else:
        cca_feats = np.zeros(1, dtype=np.float64)

    return np.concatenate([mean_band_energy, mean_coh, cca_feats]).astype(np.float64)


def _compute_wst_msc_feature(name: str) -> np.ndarray:
    X, _, fs = _load_multichannel_for_name(name)
    # Build bands
    bands = getattr(config, 'PHYSICAL_BANDS', [(0.5, 5.0), (5.0, 15.0), (15.0, 40.0)])
    # Per-channel WST-like features
    feats_wst = []
    for c in range(X.shape[1]):
        feats_wst.append(_compute_wst_features_per_channel(X[:, c], fs, bands))
    # Pad or truncate per-channel vectors to same length
    max_len = max(f.shape[0] for f in feats_wst)
    padded = []
    for fch in feats_wst:
        if fch.shape[0] < max_len:
            pad = np.zeros(max_len - fch.shape[0], dtype=np.float64)
            padded.append(np.concatenate([fch, pad]))
        else:
            padded.append(fch[:max_len])
    feats_wst = np.concatenate(padded, axis=0)

    # Multi-channel relationship features
    feats_rel = _compute_multichannel_relationship_features(X, fs, bands)

    return np.concatenate([feats_wst, feats_rel]).astype(np.float64)


def compute_feature_for_name(name: str) -> np.ndarray:
    backend = str(getattr(config, 'FEATURE_BACKEND', 'STFT') or 'STFT').upper()
    if backend == 'STFT':
        fs, f, t, z, x, y = compute_stft(name)
        Z = np.abs(z)
        x = np.squeeze(x)
        return _fixed_length_feature_from_stft(Z, x)
    # Default/new path: WST + MSC/CCA features
    return _compute_wst_msc_feature(name)


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


