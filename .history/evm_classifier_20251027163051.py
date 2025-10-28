import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import genextreme


class EVMClassifier:
    """
    A lightweight Extreme Value Machine-like open-set classifier using Weibull/GEV tails.
    For each known condition class, we fit tail distribution to distances to its class mean.
    During predict, compute distance to each class mean, convert to tail probability, pick max.
    If max prob < threshold, mark as unknown.
    """
    def __init__(self, tail_frac: float = 0.2, reject_threshold: float = 0.5, metric: str = 'euclidean'):
        self.tail_frac = tail_frac
        self.reject_threshold = reject_threshold
        self.metric = metric
        self.class_means: Dict[str, np.ndarray] = {}
        self.class_tail_params: Dict[str, Tuple[float, float, float]] = {}

    @staticmethod
    def _pairwise_distance(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        diff = X - mu.reshape(1, -1)
        return np.sqrt(np.sum(diff * diff, axis=1))

    def fit(self, class_to_features: Dict[str, np.ndarray]) -> None:
        self.class_means.clear()
        self.class_tail_params.clear()

        for cls, F in class_to_features.items():
            mu = np.mean(F, axis=0)
            dists = self._pairwise_distance(F, mu)
            # Fit tail on largest distances
            k = max(3, int(np.ceil(len(dists) * self.tail_frac)))
            tail = np.sort(dists)[-k:]
            # Fit GEV to tail
            c, loc, scale = genextreme.fit(tail)
            self.class_means[cls] = mu
            self.class_tail_params[cls] = (c, loc, scale)

    def score_one(self, x: np.ndarray) -> Dict[str, float]:
        scores = {}
        for cls, mu in self.class_means.items():
            c, loc, scale = self.class_tail_params[cls]
            d = float(np.linalg.norm(x - mu))
            # Survival function: higher is more likely within class support
            p = float(1.0 - genextreme.cdf(d, c, loc=loc, scale=scale))
            scores[cls] = p
        return scores

    def predict(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        scores = self.score_one(x)
        if len(scores) == 0:
            return 'Unknown', 0.0, scores
        cls_hat = max(scores, key=scores.get)
        max_p = scores[cls_hat]
        if max_p < self.reject_threshold:
            return 'Unknown', max_p, scores
        return cls_hat, max_p, scores


