import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from scipy.special import logsumexp

from feature_library import build_features_for_condition, compute_feature_for_name, extract_condition_prefix
from STFT import compute_stft
from evm_classifier import EVMClassifier
import config
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


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
        # Batch alignment state
        self.align_mode: str = str(getattr(config, 'FEATURE_ALIGN', 'NONE') or 'NONE').upper()
        self.reference_batch: str = str(getattr(config, 'REFERENCE_BATCH_SUFFIX', '01') or '01')
        # For CORAL: map batch suffix -> (A, b) so that x' = (x - b0) @ A + b1; we pack (A, b0, b1)
        self._coral_params: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # For WHITEN: (A, b) with x' = (x - b) @ A
        self._whiten_A: Optional[np.ndarray] = None
        self._whiten_b: Optional[np.ndarray] = None
        # Standardization and dimensionality reduction
        self._standardize: bool = bool(getattr(config, 'FEATURE_STANDARDIZE', True))
        self._scaler: Optional[StandardScaler] = None
        self._dimred: str = str(getattr(config, 'FEATURE_DIM_REDUCTION', 'PCA') or 'NONE').upper()
        self._dr_n_components: int = int(getattr(config, 'DR_N_COMPONENTS', 32))
        self._pca: Optional[PCA] = None
        self._pls: Optional[PLSRegression] = None

    @staticmethod
    def _suffix_of(name: str) -> str:
        parts = name.split('_')
        return parts[1] if len(parts) > 1 else ''

    @staticmethod
    def _cov_eig_sqrt(C: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Eigen decomposition with stabilization
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        Winv_sqrt = np.diag(1.0 / np.sqrt(w))
        Wsqrt = np.diag(np.sqrt(w))
        return V, Wsqrt, Winv_sqrt

    def _fit_alignment(self, samples: List[Tuple[str, np.ndarray]]) -> None:
        if self.align_mode not in ['CORAL', 'WHITEN']:
            return
        if len(samples) == 0:
            return
        X = np.vstack([x for _, x in samples])
        D = X.shape[1]
        eps = float(getattr(config, 'ALIGN_EPS', 1e-6))
        if self.align_mode == 'WHITEN':
            mu = np.mean(X, axis=0)
            Xc = X - mu
            C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
            V, _, Winv_sqrt = self._cov_eig_sqrt(C, eps)
            A = V @ Winv_sqrt @ V.T
            self._whiten_A = A.astype(np.float64)
            self._whiten_b = mu.astype(np.float64)
            return

        # CORAL: compute reference batch stats
        # Group by batch suffix
        batch_to_idx: Dict[str, List[int]] = {}
        for i, (name, _) in enumerate(samples):
            sfx = self._suffix_of(name)
            batch_to_idx.setdefault(sfx, []).append(i)
        # Reference stats
        ref_idx = batch_to_idx.get(self.reference_batch, [])
        if len(ref_idx) < 2:
            # Fallback to pooled as reference
            ref_idx = list(range(len(samples)))
        Xref = X[ref_idx]
        mu_ref = np.mean(Xref, axis=0)
        Xrefc = Xref - mu_ref
        Cref = (Xrefc.T @ Xrefc) / max(1, (Xrefc.shape[0] - 1))
        Vt, Wt_sqrt, Wt_inv_sqrt = self._cov_eig_sqrt(Cref, eps)
        # Target color matrix sqrt(Ct)
        Ct_sqrt = Vt @ Wt_sqrt @ Vt.T
        # Precompute whiten of source batch and color to target per batch
        self._coral_params.clear()
        for sfx, idxs in batch_to_idx.items():
            Xs = X[idxs]
            mu_s = np.mean(Xs, axis=0)
            Xsc = Xs - mu_s
            Cs = (Xsc.T @ Xsc) / max(1, (Xsc.shape[0] - 1))
            Vs, _, Vs_inv_sqrt = self._cov_eig_sqrt(Cs, eps)
            Cs_inv_sqrt = Vs @ Vs_inv_sqrt @ Vs.T
            # A maps centered src to target-colored space: A = Cs^{-1/2} Ct^{1/2}
            A = Cs_inv_sqrt @ Ct_sqrt
            b0 = mu_s
            b1 = mu_ref
            self._coral_params[sfx] = (A.astype(np.float64), b0.astype(np.float64), b1.astype(np.float64))

    def _apply_alignment(self, name: str, x: np.ndarray) -> np.ndarray:
        if self.align_mode == 'WHITEN' and self._whiten_A is not None and self._whiten_b is not None:
            return (x - self._whiten_b) @ self._whiten_A
        if self.align_mode == 'CORAL' and len(self._coral_params) > 0:
            sfx = self._suffix_of(name)
            params = self._coral_params.get(sfx)
            if params is None:
                # If unseen batch at test time, map to reference via whitening to pooled (approx): use first available A,b
                params = next(iter(self._coral_params.values()))
            A, b0, b1 = params
            return (x - b0) @ A + b1
        return x

    def fit_groups(self, trainlist: List[List[str]]) -> None:
        class_to_features: Dict[str, np.ndarray] = {}
        # Collect all (name, feat) for alignment
        all_samples: List[Tuple[str, np.ndarray]] = []
        name_to_feat: Dict[str, np.ndarray] = {}
        for group in trainlist:
            prefix = extract_condition_prefix(group)
            feats = []
            for name in group:
                f = compute_feature_for_name(name)
                name_to_feat[name] = f
                all_samples.append((name, f))
                feats.append(f.reshape(1, -1))
            class_to_features[prefix] = np.vstack(feats)

        # Fit alignment using training samples
        self._fit_alignment(all_samples)

        # Apply alignment to all features before fitting EVM
        if self.align_mode in ['CORAL', 'WHITEN']:
            for cls, F in class_to_features.items():
                # Need the corresponding names to apply per-batch transforms if CORAL
                # Find names in trainlist for this class in the same order as rows
                names_for_cls: List[str] = []
                for group in trainlist:
                    if extract_condition_prefix(group) == cls:
                        names_for_cls = group
                        break
                if len(names_for_cls) == F.shape[0]:
                    F_aligned = []
                    for i, name in enumerate(names_for_cls):
                        x = F[i]
                        F_aligned.append(self._apply_alignment(name, x))
                    class_to_features[cls] = np.vstack(F_aligned)
                else:
                    # Fallback: apply a single transform if available (e.g., WHITEN)
                    if self.align_mode == 'WHITEN' and self._whiten_A is not None and self._whiten_b is not None:
                        class_to_features[cls] = (F - self._whiten_b.reshape(1, -1)) @ self._whiten_A

        # Optional standardization (global, across all samples)
        if self._standardize:
            X_all = np.vstack(list(class_to_features.values()))
            self._scaler = StandardScaler(with_mean=True, with_std=True)
            self._scaler.fit(X_all)
            for cls in list(class_to_features.keys()):
                class_to_features[cls] = self._scaler.transform(class_to_features[cls])

        # Optional dimensionality reduction
        if self._dimred in ['PCA', 'PLS']:
            X_all = np.vstack(list(class_to_features.values()))
            y_all = []
            cls_names = list(class_to_features.keys())
            offset = 0
            for i, cls in enumerate(cls_names):
                n = class_to_features[cls].shape[0]
                y_all.extend([i] * n)
                offset += n
            y_all = np.asarray(y_all, dtype=np.int64)

            n_samples = X_all.shape[0]
            n_features = X_all.shape[1]
            # cap by both n_samples and n_features
            n_comp = max(1, min(self._dr_n_components, n_samples, n_features))
            if self._dimred == 'PCA':
                self._pca = PCA(n_components=n_comp, random_state=getattr(config, 'RANDOM_STATE', 0))
                self._pca.fit(X_all)
                for cls in list(class_to_features.keys()):
                    class_to_features[cls] = self._pca.transform(class_to_features[cls])
            else:
                # Encode labels as continuous targets for PLS; one-vs-all encoding
                Y = np.zeros((X_all.shape[0], len(cls_names)), dtype=float)
                for i, cls in enumerate(cls_names):
                    start = sum(class_to_features[c].shape[0] for c in cls_names[:i])
                    end = start + class_to_features[cls].shape[0]
                    Y[start:end, i] = 1.0
                n_targets = Y.shape[1]
                n_comp_pls = max(1, min(n_comp, n_targets, n_samples - 1, n_features))
                self._pls = PLSRegression(n_components=n_comp_pls, scale=False)
                self._pls.fit(X_all, Y)
                # Transform per class
                # Need to recompute class_to_features in the same order to preserve rows; rebuild from X_all
                transformed = self._pls.transform(X_all)
                idx = 0
                new_map: Dict[str, np.ndarray] = {}
                for cls in cls_names:
                    n = class_to_features[cls].shape[0]
                    new_map[cls] = transformed[idx:idx + n]
                    idx += n
                class_to_features = new_map

        self.evm.fit(class_to_features)

    def predict_on_name(self, name: str) -> Tuple[str, float, Dict[str, float]]:
        feat = compute_feature_for_name(name)
        # Apply the same alignment as training
        feat = self._apply_alignment(name, feat)
        # Standardize
        if self._standardize and self._scaler is not None:
            feat = self._scaler.transform(feat.reshape(1, -1)).reshape(-1)
        # Dimensionality reduction
        if self._pca is not None:
            feat = self._pca.transform(feat.reshape(1, -1)).reshape(-1)
        elif self._pls is not None:
            feat = self._pls.transform(feat.reshape(1, -1)).reshape(-1)
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


def get_recognizer(strategy: str, evm_tail_frac: float = 0.5, evm_reject_threshold: float = 0.5,
                   hmm_n_states: int = 3, hmm_cov_reg: float = 1e-6, hmm_n_iter: int = 15, hmm_glrt_quantile: float = 0.1,
                   random_state: int = 0) -> ConditionRecognizer:
    s = (strategy or '').lower()
    if s in ['hmm', 'hsmm']:
        return HMMGLRTRecognizer(n_states=hmm_n_states, cov_reg=hmm_cov_reg, n_iter=hmm_n_iter,
                                 glrt_quantile=hmm_glrt_quantile, random_state=random_state)
    # default to EVM
    return EVMRecognizer(tail_frac=evm_tail_frac, reject_threshold=evm_reject_threshold)


