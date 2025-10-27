import os
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

    @staticmethod
    def _pairwise_distance(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        diff = X - mu.reshape(1, -1)
        return np.sqrt(np.sum(diff * diff, axis=1))

    def fit(self, class_to_features: Dict[str, np.ndarray]) -> None:
        self.class_means.clear()
        self.class_tail_params.clear()
        self.class_dist_thresh.clear()

        for cls, F in class_to_features.items():
            mu = np.mean(F, axis=0)
            dists = self._pairwise_distance(F, mu)
            # Fit Weibull on largest distances (tail)
            n = len(dists)
            k = max(5, int(np.ceil(n * self.tail_frac))) if n >= 5 else n
            tail = np.sort(dists)[-k:]
            # Guard small or degenerate tail by adding tiny jitter
            if k <= 2:
                tail = np.concatenate([tail, tail])
            tail = tail + 1e-6
            c, loc, scale = weibull_min.fit(tail, floc=0)
            self.class_means[cls] = mu
            self.class_tail_params[cls] = (c, loc, scale)
            # 95th percentile as distance threshold for unknown gating
            self.class_dist_thresh[cls] = float(np.quantile(dists, 0.95) if n > 2 else np.max(dists) * 1.05)

    def score_one(self, x: np.ndarray) -> Dict[str, float]:
        scores = {}
        for cls, mu in self.class_means.items():
            c, loc, scale = self.class_tail_params[cls]
            d = float(np.linalg.norm(x - mu))
            # Weibull survival: prob of being within the class support
            sf = float(weibull_min.sf(d, c, loc=loc, scale=scale))
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
        mu = self.class_means.get(cls_hat)
        d_hat = float(np.linalg.norm(x - mu)) if mu is not None else float('inf')
        dist_thresh = self.class_dist_thresh.get(cls_hat, float('inf'))
        if (max_p < self.reject_threshold) or (d_hat > self.unknown_margin * dist_thresh):
            return 'Unknown', max_p, scores
        return cls_hat, max_p, scores


