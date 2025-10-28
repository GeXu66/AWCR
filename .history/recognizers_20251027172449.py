import os
import numpy as np
import scipy.io as scio
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from scipy.special import logsumexp

from feature_library import build_features_for_condition, compute_feature_for_name, extract_condition_prefix
from STFT import compute_stft
from evm_classifier import EVMClassifier


class ConditionRecognizer(ABC):
    @abstractmethod
    def fit_groups(self, trainlist: List[List[str]]) -> None:
        ...

    @abstractmethod
    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        ...


class EVMRecognizer(ConditionRecognizer):
    def __init__(self, tail_frac: float = 0.5, reject_threshold: float = 0.5) -> None:
        self.evm = EVMClassifier(tail_frac=tail_frac, reject_threshold=reject_threshold)

    def fit_groups(self, trainlist: List[List[str]]) -> None:
        class_to_features: Dict[str, np.ndarray] = {}
        for group in trainlist:
            prefix = extract_condition_prefix(group)
            F = build_features_for_condition(group)
            class_to_features[prefix] = F
        self.evm.fit(class_to_features)

    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        feat = compute_feature_for_name(name)
        cls_hat, prob, scores = self.evm.predict(feat)
        return cls_hat, prob, scores


class GaussianHMM:
    def __init__(self, n_states: int = 3, cov_reg: float = 1e-6, n_iter: int = 15, random_state: int = 0) -> None:
        self.n_states = n_states
        self.cov_reg = cov_reg
        self.n_iter = n_iter
        self.random_state = np.random.RandomState(random_state)
        self.pi: np.ndarray = np.empty((0,))
        self.A: np.ndarray = np.empty((0, 0))
        self.means: np.ndarray = np.empty((0, 0))
        self.vars: np.ndarray = np.empty((0, 0))  # diagonal variances (K x D)

    def _init_params(self, X_concat: np.ndarray) -> None:
        K = self.n_states
        D = X_concat.shape[1]
        # Initialize means by random frames
        idx = self.random_state.choice(X_concat.shape[0], size=K, replace=False)
        self.means = X_concat[idx].copy()
        # Initialize diagonal variances from global variance
        global_var = np.var(X_concat, axis=0) + self.cov_reg
        self.vars = np.tile(global_var.reshape(1, -1), (K, 1))
        # Start with near-identity transitions to encourage durations
        self.A = np.full((K, K), 1.0 / K, dtype=float)
        for k in range(K):
            self.A[k, :] = (1.0 - 0.9) / (K - 1)
            self.A[k, k] = 0.9
        # Uniform initial state
        self.pi = np.full(K, 1.0 / K, dtype=float)

    @staticmethod
    def _log_gaussian_diag(X: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        # X: (T,D), means: (K,D), vars: (K,D)
        T, D = X.shape
        K = means.shape[0]
        # Broadcast to (T,K,D)
        X_exp = X[:, None, :]
        means_exp = means[None, :, :]
        vars_exp = vars_[None, :, :]
        log_det = 0.5 * np.sum(np.log(2.0 * np.pi * vars_), axis=1)  # (K,)
        quad = 0.5 * np.sum(((X_exp - means_exp) ** 2) / vars_exp, axis=2)  # (T,K)
        return -(quad + log_det[None, :])  # (T,K)

    def _forward_backward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        K = self.n_states
        T = X.shape[0]
        log_B = self._log_gaussian_diag(X, self.means, self.vars)  # (T,K)
        log_pi = np.log(self.pi + 1e-16)  # (K,)
        log_A = np.log(self.A + 1e-16)    # (K,K)

        # Forward
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            # logsumexp over previous states
            log_alpha[t] = log_B[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

        log_likelihood = float(logsumexp(log_alpha[-1]))

        # Backward
        log_beta = np.empty((T, K))
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(log_A + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1)

        # Posteriors
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)

        # Xi expectations
        xi = np.empty((T - 1, K, K))
        for t in range(T - 1):
            log_xi_t = (log_alpha[t][:, None] + log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :]) - log_likelihood
            xi[t] = np.exp(log_xi_t)

        return gamma, xi, log_likelihood, log_B

    def fit(self, sequences: List[np.ndarray]) -> None:
        # Concatenate all frames for initialization
        X_concat = np.vstack(sequences)
        self._init_params(X_concat)

        K = self.n_states
        D = X_concat.shape[1]
        for _ in range(self.n_iter):
            # Accumulators
            pi_acc = np.zeros(K)
            A_num = np.zeros((K, K))
            A_den = np.zeros(K)
            means_num = np.zeros((K, D))
            means_den = np.zeros(K)
            vars_num = np.zeros((K, D))

            for X in sequences:
                gamma, xi, _, _ = self._forward_backward(X)
                pi_acc += gamma[0]
                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1], axis=0)
                means_num += gamma.T @ X
                means_den += np.sum(gamma, axis=0)

            # Update parameters with regularization to avoid degenerate states
            self.pi = (pi_acc + 1e-6)
            self.pi /= np.sum(self.pi)

            A = A_num + 1e-6
            A /= (A_den[:, None] + 1e-12)
            self.A = A

            means = means_num / (means_den[:, None] + 1e-12)
            self.means = means

            # Variances
            for X in sequences:
                gamma, _, _, _ = self._forward_backward(X)
                diff = X[:, None, :] - self.means[None, :, :]
                vars_num += np.sum(gamma[:, :, None] * (diff ** 2), axis=0)
            vars_ = vars_num / (means_den[:, None] + 1e-12)
            vars_ = np.maximum(vars_, self.cov_reg)
            self.vars = vars_

    def score(self, X: np.ndarray) -> float:
        _, _, log_likelihood, _ = self._forward_backward(X)
        return log_likelihood


def _sequence_features_from_stft(abs_z: np.ndarray) -> np.ndarray:
    # Per-timeframe features: mean and std across frequency bins
    # abs_z: (F, T)
    mean_t = np.mean(abs_z, axis=0)
    std_t = np.std(abs_z, axis=0)
    seq = np.stack([mean_t, std_t], axis=1)  # (T,2)
    return seq.astype(np.float64)


class HMMGLRTRecognizer(ConditionRecognizer):
    def __init__(self, n_states: int = 3, cov_reg: float = 1e-6, n_iter: int = 15, glrt_quantile: float = 0.1, random_state: int = 0) -> None:
        self.n_states = n_states
        self.cov_reg = cov_reg
        self.n_iter = n_iter
        self.glrt_quantile = glrt_quantile
        self.random_state = random_state
        self.class_models: Dict[str, GaussianHMM] = {}
        self.background_model: GaussianHMM = GaussianHMM(n_states=n_states, cov_reg=cov_reg, n_iter=n_iter, random_state=random_state)
        self.reject_threshold: float = 0.0

    @staticmethod
    def _seq_for_name(name: str) -> np.ndarray:
        fs, f, t, z, x, y = compute_stft(name)
        Z = np.abs(z)
        return _sequence_features_from_stft(Z)

    def fit_groups(self, trainlist: List[List[str]]) -> None:
        # Build sequences per class
        class_to_sequences: Dict[str, List[np.ndarray]] = {}
        all_sequences: List[np.ndarray] = []
        per_sample_records: List[Tuple[str, np.ndarray]] = []
        for group in trainlist:
            prefix = extract_condition_prefix(group)
            seqs = []
            for name in group:
                seq = self._seq_for_name(name)
                seqs.append(seq)
                all_sequences.append(seq)
                per_sample_records.append((prefix, seq))
            class_to_sequences[prefix] = seqs

        # Train per-class HMMs
        self.class_models.clear()
        for cls, seqs in class_to_sequences.items():
            hmm = GaussianHMM(n_states=self.n_states, cov_reg=self.cov_reg, n_iter=self.n_iter, random_state=self.random_state)
            hmm.fit(seqs)
            self.class_models[cls] = hmm

        # Train background model on all data
        self.background_model = GaussianHMM(n_states=self.n_states, cov_reg=self.cov_reg, n_iter=self.n_iter, random_state=self.random_state)
        self.background_model.fit(all_sequences)

        # Calibrate GLRT threshold on training data
        glr_values: List[float] = []
        for cls, seq in per_sample_records:
            ll_cls = self.class_models[cls].score(seq)
            ll_bg = self.background_model.score(seq)
            glr_values.append(ll_cls - ll_bg)
        if len(glr_values) > 0:
            self.reject_threshold = float(np.quantile(glr_values, self.glrt_quantile))
        else:
            self.reject_threshold = 0.0

    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        seq = self._seq_for_name(name)
        # Compute per-class log-likelihoods
        loglikes: Dict[str, float] = {}
        for cls, hmm in self.class_models.items():
            loglikes[cls] = hmm.score(seq)
        if len(loglikes) == 0:
            return 'Unknown', 0.0, {}

        # Background model
        ll_bg = self.background_model.score(seq)
        # GLR for best class
        best_cls = max(loglikes, key=loglikes.get)
        glr = loglikes[best_cls] - ll_bg

        # Convert loglikes to probabilities with softmax for interpretability
        ll_values = np.array(list(loglikes.values()), dtype=float)
        ll_keys = list(loglikes.keys())
        ll_shift = ll_values - np.max(ll_values)
        probs = np.exp(ll_shift)
        probs /= np.sum(probs)
        scores = {k: float(p) for k, p in zip(ll_keys, probs)}

        if glr < self.reject_threshold:
            return 'Unknown', float(np.max(probs)), scores
        return best_cls, float(scores[best_cls]), scores


def _maybe_downsample(X: np.ndarray, stride: int) -> np.ndarray:
    if stride is None or stride <= 1:
        return X
    return X[::stride]


def _center_per_channel(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=0, keepdims=True)
    return X - mu


def _dtw_distance_multivariate(X: np.ndarray, Y: np.ndarray, window: Optional[int]) -> float:
    """
    Dynamic Time Warping distance between two multivariate time series.
    Local cost is Euclidean distance between D-dim vectors.
    Returns path-normalized distance.
    """
    T, D = X.shape
    U, D2 = Y.shape
    if D != D2:
        raise ValueError('dimension mismatch in DTW sequences')
    if window is None:
        window = max(T, U)
    window = max(window, abs(T - U))

    inf = 1e18
    dp = np.full((T + 1, U + 1), inf, dtype=float)
    dp[0, 0] = 0.0
    for t in range(1, T + 1):
        u_start = max(1, t - window)
        u_end = min(U, t + window)
        x_t = X[t - 1]
        for u in range(u_start, u_end + 1):
            # Euclidean local cost
            diff = x_t - Y[u - 1]
            cost = float(np.sqrt(np.dot(diff, diff)))
            dp[t, u] = cost + min(dp[t - 1, u], dp[t, u - 1], dp[t - 1, u - 1])
    dist = dp[T, U]
    # Normalize by approximate path length to reduce linear dependence on duration
    norm = T + U
    return dist / max(1, norm)


class DTWRecognizer(ConditionRecognizer):
    def __init__(self,
                 channel_in: List[int],
                 mean_flag: bool = True,
                 window_ratio: Optional[float] = 0.1,
                 downsample: int = 4,
                 reject_quantile: float = 0.9,
                 margin: float = 1.2,
                 data_root: str = '../Data/matdata') -> None:
        self.channel_in = channel_in
        self.mean_flag = mean_flag
        self.window_ratio = window_ratio
        self.downsample = max(1, int(downsample))
        self.reject_quantile = reject_quantile
        self.margin = margin
        self.data_root = data_root

        self.class_to_sequences: Dict[str, List[np.ndarray]] = {}
        self.reject_threshold: float = np.inf

    def _read_sequence(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_root, name + '.mat')
        key = os.path.basename(path).split('.')[-2]
        data = scio.loadmat(path)[key]
        X = data[:, self.channel_in].astype(np.float64)
        if self.mean_flag:
            X = _center_per_channel(X)
        X = _maybe_downsample(X, self.downsample)
        return X

    def _window_size(self, T: int, U: int) -> Optional[int]:
        if self.window_ratio is None:
            return None
        w = int(self.window_ratio * min(T, U))
        return max(0, w)

    def _min_dtw_to_class(self, seq: np.ndarray, class_seqs: List[np.ndarray]) -> float:
        best = np.inf
        for ref in class_seqs:
            w = self._window_size(len(seq), len(ref))
            d = _dtw_distance_multivariate(seq, ref, window=w)
            if d < best:
                best = d
        return best

    def fit_groups(self, trainlist: List[List[str]]) -> None:
        self.class_to_sequences.clear()
        # Load sequences
        for group in trainlist:
            if len(group) == 0:
                continue
            prefix = extract_condition_prefix(group)
            seqs: List[np.ndarray] = []
            for name in group:
                X = self._read_sequence(name)
                seqs.append(X)
            self.class_to_sequences[prefix] = seqs

        # Calibrate open-set threshold from within-class nearest neighbor distances (leave-one-out)
        intra_nearest: List[float] = []
        for cls, seqs in self.class_to_sequences.items():
            n = len(seqs)
            if n <= 1:
                continue
            for i in range(n):
                best = np.inf
                for j in range(n):
                    if i == j:
                        continue
                    w = self._window_size(len(seqs[i]), len(seqs[j]))
                    d = _dtw_distance_multivariate(seqs[i], seqs[j], window=w)
                    if d < best:
                        best = d
                if np.isfinite(best):
                    intra_nearest.append(best)
        if len(intra_nearest) == 0:
            # fallback threshold if only one sample per class; use median inter-class distance among exemplars
            inter: List[float] = []
            keys = list(self.class_to_sequences.keys())
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    for xa in self.class_to_sequences[keys[a]]:
                        for xb in self.class_to_sequences[keys[b]]:
                            w = self._window_size(len(xa), len(xb))
                            inter.append(_dtw_distance_multivariate(xa, xb, window=w))
            base = float(np.median(inter)) if len(inter) > 0 else 1.0
            self.reject_threshold = base * self.margin
        else:
            base = float(np.quantile(intra_nearest, self.reject_quantile))
            self.reject_threshold = base * self.margin

    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        if len(self.class_to_sequences) == 0:
            return 'Unknown', 0.0, {}
        seq = self._read_sequence(name)
        class_dists: Dict[str, float] = {}
        for cls, seqs in self.class_to_sequences.items():
            class_dists[cls] = self._min_dtw_to_class(seq, seqs)
        # Select best (smallest distance)
        best_cls = min(class_dists, key=class_dists.get)
        best_dist = class_dists[best_cls]

        # Convert distances to normalized inverse weights for interpretability
        d_vals = np.array(list(class_dists.values()), dtype=float)
        # shift to avoid division by zero
        eps = max(1e-12, float(np.min(d_vals)) * 1e-6)
        inv = 1.0 / (d_vals + eps)
        probs = inv / np.sum(inv)
        scores = {k: float(p) for k, p in zip(class_dists.keys(), probs)}

        if best_dist > self.reject_threshold:
            return 'Unknown', float(scores.get(best_cls, 0.0)), scores
        return best_cls, float(scores[best_cls]), scores


def get_recognizer(strategy: str, evm_tail_frac: float = 0.5, evm_reject_threshold: float = 0.5,
                   hmm_n_states: int = 3, hmm_cov_reg: float = 1e-6, hmm_n_iter: int = 15, hmm_glrt_quantile: float = 0.1,
                   random_state: int = 0,
                   # DTW specific
                   dtw_window_ratio: Optional[float] = 0.1, dtw_downsample: int = 4,
                   dtw_reject_quantile: float = 0.9, dtw_margin: float = 1.2,
                   channel_in: Optional[List[int]] = None, mean_flag: bool = True) -> ConditionRecognizer:
    s = (strategy or '').lower()
    if s in ['hmm', 'hsmm']:
        return HMMGLRTRecognizer(n_states=hmm_n_states, cov_reg=hmm_cov_reg, n_iter=hmm_n_iter,
                                 glrt_quantile=hmm_glrt_quantile, random_state=random_state)
    if s in ['dtw']:
        if channel_in is None:
            raise ValueError('DTW strategy requires channel_in to be provided')
        return DTWRecognizer(channel_in=channel_in, mean_flag=mean_flag,
                             window_ratio=dtw_window_ratio, downsample=dtw_downsample,
                             reject_quantile=dtw_reject_quantile, margin=dtw_margin)
    # default to EVM
    return EVMRecognizer(tail_frac=evm_tail_frac, reject_threshold=evm_reject_threshold)


