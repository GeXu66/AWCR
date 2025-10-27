import os
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import weibull_min


class EVMClassifier:
    """
    A lightweight Extreme Value Machine-like open-set classifier using Weibull/GEV tails.
    For each known condition class, we fit tail distribution to distances to its class mean.
    During predict, compute distance to each class mean, convert to tail probability, pick max.
    If max prob < threshold, mark as unknown.
    """
    def __init__(self,
                 tail_frac: float = 0.3,
                 reject_threshold: float = 0.35,
                 metric: str = 'euclidean',
                 temp: float = 1.0,
                 eps: float = 1e-6,
                 unknown_margin: float = 1.1):
        self.tail_frac = tail_frac
        self.reject_threshold = reject_threshold
        self.metric = metric
        self.temp = temp
        self.eps = eps
        self.unknown_margin = unknown_margin
        self.class_means: Dict[str, np.ndarray] = {}
        # store Weibull params per class: (shape, loc, scale)
        self.class_tail_params: Dict[str, Tuple[float, float, float]] = {}
        # class distance threshold (e.g., 95th percentile of training distances)
        self.class_dist_thresh: Dict[str, float] = {}
        # feature normalization
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None

    @staticmethod
    def _pairwise_distance(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        diff = X - mu.reshape(1, -1)
        return np.sqrt(np.sum(diff * diff, axis=1))

    def fit(self, class_to_features: Dict[str, np.ndarray]) -> None:
        self.class_means.clear()
        self.class_tail_params.clear()
        self.class_dist_thresh.clear()

        # global normalization
        allF = np.vstack([F for F in class_to_features.values() if F is not None and len(F) > 0])
        self.feat_mean = np.mean(allF, axis=0)
        std = np.std(allF, axis=0)
        std[std < self.eps] = 1.0
        self.feat_std = std

        rng = np.random.RandomState(42)

        for cls, F in class_to_features.items():
            Fz = (F - self.feat_mean) / self.feat_std
            mu = np.mean(Fz, axis=0)
            dists = self._pairwise_distance(Fz, mu)
            # Fit Weibull on largest distances (tail)
            n = len(dists)
            k = max(5, int(np.ceil(n * self.tail_frac))) if n >= 5 else n
            tail = np.sort(dists)[-k:]
            # Guard small or degenerate tail by adding tiny jitter
            if k <= 2:
                tail = np.concatenate([tail, tail])
            if np.std(tail) < self.eps:
                tail = tail + (self.eps * (1.0 + 0.01 * rng.randn(*tail.shape)))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    c, loc, scale = weibull_min.fit(tail, floc=0)
            except Exception:
                c, loc, scale = 1.5, 0.0, float(np.median(tail) + self.eps)
            # clamp params
            if not np.isfinite(c) or c <= 0.2:
                c = 1.5
            if not np.isfinite(scale) or scale <= self.eps:
                scale = float(np.median(tail) + self.eps)
            loc = 0.0
            self.class_means[cls] = mu
            self.class_tail_params[cls] = (float(c), float(loc), float(scale))
            # 95th percentile as distance threshold for unknown gating
            self.class_dist_thresh[cls] = float(np.quantile(dists, 0.95) if n > 2 else np.max(dists) * 1.05)

    def score_one(self, x: np.ndarray) -> Dict[str, float]:
        # normalize input feature
        if self.feat_mean is not None and self.feat_std is not None:
            xz = (x - self.feat_mean) / self.feat_std
        else:
            xz = x
        scores = {}
        for cls, mu in self.class_means.items():
            c, loc, scale = self.class_tail_params[cls]
            d = float(np.linalg.norm(xz - mu))
            d = max(d, self.eps)
            # Stable log survival: log(SF) = - (d/scale)^c
            y = d / max(scale, self.eps)
            # compute exponent safely via logs
            logy = float(np.log(max(y, self.eps)))
            expo = c * logy
            # if exponent huge, clamp
            if expo > 60.0:
                logsf = -60.0
            elif expo < -50.0:
                logsf = 0.0
            else:
                logsf = -float(np.exp(expo))
            logsf = float(np.clip(logsf, -60.0, 0.0))
            sf = float(np.exp(logsf))
            scores[cls] = sf
        # Temperature scaling and epsilon smoothing, normalized to sum to 1
        keys = list(scores.keys())
        vals = np.array([scores[k] for k in keys], dtype=float)
        vals = np.power(vals + self.eps, 1.0 / max(self.temp, 1e-6))
        vals = vals / (np.sum(vals) + self.eps * len(vals))
        return {k: float(v) for k, v in zip(keys, vals)}

    def predict(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        scores = self.score_one(x)
        if len(scores) == 0:
            return 'Unknown', 0.0, scores
        cls_hat = max(scores, key=scores.get)
        max_p = scores[cls_hat]
        # additional gating by distance threshold to reduce false positives
        if self.feat_mean is not None and self.feat_std is not None:
            xz = (x - self.feat_mean) / self.feat_std
        else:
            xz = x
        mu = self.class_means.get(cls_hat)
        d_hat = float(np.linalg.norm(xz - mu)) if mu is not None else float('inf')
        dist_thresh = self.class_dist_thresh.get(cls_hat, float('inf'))
        if (max_p < self.reject_threshold) or (d_hat > self.unknown_margin * dist_thresh):
            return 'Unknown', max_p, scores
        return cls_hat, max_p, scores


