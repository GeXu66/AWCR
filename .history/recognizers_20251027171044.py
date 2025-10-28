import numpy as np
from typing import Dict, List, Tuple
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
    def __init__(self,
                 n_states: int = 3,
                 cov_reg: float = 1e-6,
                 n_iter: int = 15,
                 glrt_quantile: float = 0.1,
                 random_state: int = 0,
                 # feature options
                 feature_mode: str = 'stft',  # 'stft' | 'stft_bands' | 'raw' | 'raw_win'
                 num_bands: int = 4,
                 raw_win: int = 32,
                 raw_hop: int = 16,
                 standardize: bool = True,
                 # training options
                 restarts: int = 1,
                 self_transition: float = 0.9,
                 use_avg_ll: bool = True) -> None:
        self.n_states = n_states
        self.cov_reg = cov_reg
        self.n_iter = n_iter
        self.glrt_quantile = glrt_quantile
        self.random_state = random_state
        self.feature_mode = feature_mode
        self.num_bands = num_bands
        self.raw_win = raw_win
        self.raw_hop = raw_hop
        self.standardize = standardize
        self.restarts = max(1, restarts)
        self.self_transition = np.clip(self_transition, 0.5, 0.999)
        self.use_avg_ll = use_avg_ll
        self.class_models: Dict[str, GaussianHMM] = {}
        self.background_model: GaussianHMM = GaussianHMM(n_states=n_states, cov_reg=cov_reg, n_iter=n_iter, random_state=random_state)
        self.class_thresholds: Dict[str, float] = {}
        self.feature_mean: np.ndarray = np.empty((0,))
        self.feature_std: np.ndarray = np.empty((0,))

    def _build_seq(self, name: str) -> np.ndarray:
        fs, f, t, z, x, y = compute_stft(name)
        if self.feature_mode == 'raw':
            # z-score raw 1D signal and treat each sample as a frame
            xz = (x - np.mean(x)) / (np.std(x) + 1e-8)
            seq = xz.reshape(-1, 1)
            return seq.astype(np.float64)
        if self.feature_mode == 'raw_win':
            w = int(self.raw_win)
            h = int(self.raw_hop)
            T = max(0, (len(x) - w) // h + 1)
            feats = []
            for i in range(T):
                seg = x[i * h:i * h + w]
                mu = float(np.mean(seg))
                sd = float(np.std(seg))
                feats.append([mu, sd])
            if len(feats) == 0:
                feats = [[0.0, 1.0]]
            return np.asarray(feats, dtype=np.float64)
        # STFT-based
        Z = np.abs(z)
        if self.feature_mode == 'stft_bands':
            # Split frequency bins into num_bands and compute per-frame band means
            F, T = Z.shape
            k = max(1, int(self.num_bands))
            idx = np.linspace(0, F, k + 1).astype(int)
            bands = []
            for bi in range(k):
                f_start, f_end = idx[bi], idx[bi + 1]
                if f_end <= f_start:
                    f_end = min(F, f_start + 1)
                band_mean = np.mean(Z[f_start:f_end, :], axis=0)
                bands.append(band_mean)
            seq = np.stack(bands, axis=1)  # (T, k)
            return seq.astype(np.float64)
        # default 'stft': mean/std per frame
        return _sequence_features_from_stft(Z)

    def _standardize(self, seq: np.ndarray) -> np.ndarray:
        if not self.standardize or self.feature_mean.size == 0:
            return seq
        return (seq - self.feature_mean) / (self.feature_std + 1e-8)

    def _avg_ll(self, hmm: GaussianHMM, seq: np.ndarray) -> float:
        ll = hmm.score(seq)
        return float(ll / max(1, seq.shape[0])) if self.use_avg_ll else float(ll)

    def _fit_best_hmm(self, sequences: List[np.ndarray], seed_offset: int = 0) -> GaussianHMM:
        best_model: GaussianHMM = None
        best_score = -np.inf
        for r in range(self.restarts):
            hmm = GaussianHMM(n_states=self.n_states, cov_reg=self.cov_reg, n_iter=self.n_iter, random_state=self.random_state + seed_offset + r)
            # Encourage longer durations via self-transition after fit by adjusting A
            hmm.fit(sequences)
            # Adjust transitions towards higher self-transition probability
            K = hmm.A.shape[0]
            A = np.full_like(hmm.A, (1.0 - self.self_transition) / max(1, K - 1))
            np.fill_diagonal(A, self.self_transition)
            hmm.A = A
            # Evaluate
            total = 0.0
            for X in sequences:
                total += self._avg_ll(hmm, X)
            if total > best_score:
                best_score = total
                best_model = hmm
        return best_model

    def fit_groups(self, trainlist: List[List[str]]) -> None:
        # Build sequences per class
        class_to_sequences: Dict[str, List[np.ndarray]] = {}
        all_sequences: List[np.ndarray] = []
        per_sample_records: List[Tuple[str, np.ndarray]] = []
        for group in trainlist:
            prefix = extract_condition_prefix(group)
            seqs = []
            for name in group:
                seq_raw = self._build_seq(name)
                seqs.append(seq_raw)
                all_sequences.append(seq_raw)
                per_sample_records.append((prefix, seq_raw))
            class_to_sequences[prefix] = seqs

        # Compute feature standardization from all frames
        if self.standardize:
            if len(all_sequences) > 0:
                concat = np.vstack(all_sequences)
                self.feature_mean = np.mean(concat, axis=0)
                self.feature_std = np.std(concat, axis=0)
                self.feature_std[self.feature_std < 1e-6] = 1.0
        # Apply standardization
        if self.standardize:
            for cls in class_to_sequences:
                class_to_sequences[cls] = [self._standardize(X) for X in class_to_sequences[cls]]
            all_sequences = [self._standardize(X) for X in all_sequences]
            per_sample_records = [(cls, self._standardize(X)) for (cls, X) in per_sample_records]

        # Train per-class HMMs with restarts
        self.class_models.clear()
        seed_offset = 0
        for cls, seqs in class_to_sequences.items():
            model = self._fit_best_hmm(seqs, seed_offset=seed_offset)
            self.class_models[cls] = model
            seed_offset += 13

        # Train background model on all data with restarts
        self.background_model = self._fit_best_hmm(all_sequences, seed_offset=777)

        # Calibrate per-class GLRT thresholds on training data
        self.class_thresholds.clear()
        for cls, seqs in class_to_sequences.items():
            glr_values: List[float] = []
            for X in seqs:
                ll_c = self._avg_ll(self.class_models[cls], X)
                ll_b = self._avg_ll(self.background_model, X)
                glr_values.append(ll_c - ll_b)
            if len(glr_values) > 0:
                thr = float(np.quantile(glr_values, self.glrt_quantile))
            else:
                thr = 0.0
            self.class_thresholds[cls] = thr

    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        seq = self._build_seq(name)
        seq = self._standardize(seq)
        # Compute per-class average log-likelihoods
        loglikes: Dict[str, float] = {}
        for cls, hmm in self.class_models.items():
            loglikes[cls] = self._avg_ll(hmm, seq)
        if len(loglikes) == 0:
            return 'Unknown', 0.0, {}

        # Background model
        ll_bg = self._avg_ll(self.background_model, seq)
        # GLR for best class
        best_cls = max(loglikes, key=loglikes.get)
        glr = loglikes[best_cls] - ll_bg

        # Convert to probabilities with softmax for interpretability
        ll_values = np.array(list(loglikes.values()), dtype=float)
        ll_keys = list(loglikes.keys())
        ll_shift = ll_values - np.max(ll_values)
        probs = np.exp(ll_shift)
        probs /= np.sum(probs)
        scores = {k: float(p) for k, p in zip(ll_keys, probs)}

        thr = self.class_thresholds.get(best_cls, 0.0)
        if glr < thr:
            return 'Unknown', float(np.max(probs)), scores
        return best_cls, float(scores[best_cls]), scores


def get_recognizer(strategy: str, evm_tail_frac: float = 0.5, evm_reject_threshold: float = 0.5,
                   hmm_n_states: int = 3, hmm_cov_reg: float = 1e-6, hmm_n_iter: int = 15, hmm_glrt_quantile: float = 0.1,
                   random_state: int = 0,
                   hmm_feature_mode: str = 'stft', hmm_num_bands: int = 4, hmm_raw_win: int = 32, hmm_raw_hop: int = 16,
                   hmm_standardize: bool = True, hmm_restarts: int = 1, hmm_self_transition: float = 0.9, hmm_use_avg_ll: bool = True) -> ConditionRecognizer:
    s = (strategy or '').lower()
    if s in ['hmm', 'hsmm']:
        return HMMGLRTRecognizer(n_states=hmm_n_states, cov_reg=hmm_cov_reg, n_iter=hmm_n_iter,
                                 glrt_quantile=hmm_glrt_quantile, random_state=random_state,
                                 feature_mode=hmm_feature_mode, num_bands=hmm_num_bands, raw_win=hmm_raw_win, raw_hop=hmm_raw_hop,
                                 standardize=hmm_standardize, restarts=hmm_restarts, self_transition=hmm_self_transition,
                                 use_avg_ll=hmm_use_avg_ll)
    # default to EVM
    return EVMRecognizer(tail_frac=evm_tail_frac, reject_threshold=evm_reject_threshold)


