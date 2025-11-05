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
        self.class_to_features: Dict[str, np.ndarray] = {}

    def fit_groups(self, trainlist: List[List[str]]) -> None:
        class_to_features: Dict[str, np.ndarray] = {}
        for group in trainlist:
            prefix = extract_condition_prefix(group)
            F = build_features_for_condition(group)
            class_to_features[prefix] = F
        self.class_to_features = class_to_features
        self.evm.fit(self.class_to_features)

    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        feat = compute_feature_for_name(name)
        cls_hat, prob, scores = self.evm.predict(feat)
        return cls_hat, prob, scores

    def add_class_samples(self, cls: str, feats: np.ndarray) -> None:
        # feats: (N, D)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        if cls in self.class_to_features:
            self.class_to_features[cls] = np.vstack([self.class_to_features[cls], feats])
        else:
            self.class_to_features[cls] = feats
        self.evm.fit(self.class_to_features)


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


def get_recognizer(strategy: str, evm_tail_frac: float = 0.5, evm_reject_threshold: float = 0.5,
                   hmm_n_states: int = 3, hmm_cov_reg: float = 1e-6, hmm_n_iter: int = 15, hmm_glrt_quantile: float = 0.1,
                   random_state: int = 0) -> ConditionRecognizer:
    s = (strategy or '').lower()
    if s in ['hmm', 'hsmm']:
        return HMMGLRTRecognizer(n_states=hmm_n_states, cov_reg=hmm_cov_reg, n_iter=hmm_n_iter,
                                 glrt_quantile=hmm_glrt_quantile, random_state=random_state)
    # default to EVM
    return EVMRecognizer(tail_frac=evm_tail_frac, reject_threshold=evm_reject_threshold)


