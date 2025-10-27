import os
import json
import numpy as np
import scipy.io as scio
from typing import List, Tuple, Dict, Optional
from MIMOFIR import MIMOFIR, build_noncausal_design, _validate_slice


class MIMOFIRManager:
    def __init__(self,
                 past_order: int,
                 future_order: int,
                 channel_in: List[int],
                 channel_out: List[int],
                 mean_flag: bool,
                 models_dir: str = 'models'):
        self.past_order = past_order
        self.future_order = future_order
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.mean_flag = mean_flag
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def _model_path(self, condition_name: str) -> str:
        return os.path.join(self.models_dir, f'{condition_name}.mat')

    def exists(self, condition_name: str) -> bool:
        return os.path.exists(self._model_path(condition_name))

    def save(self, condition_name: str, Theta: np.ndarray) -> None:
        scio.savemat(self._model_path(condition_name), {
            'Theta': Theta,
            'channel_in': np.array(self.channel_in).reshape(-1, 1),
            'channel_out': np.array(self.channel_out).reshape(-1, 1),
            'past_order': self.past_order,
            'future_order': self.future_order,
            'mean_flag': int(self.mean_flag),
        })

    def load(self, condition_name: str) -> Optional[np.ndarray]:
        path = self._model_path(condition_name)
        if not os.path.exists(path):
            return None
        data = scio.loadmat(path)
        return data.get('Theta')

    def train_from_files(self, condition_name: str, train_names: List[str], piece: str = 'all') -> np.ndarray:
        model = MIMOFIR(self.past_order, self.future_order, self.channel_in, self.channel_out, self.mean_flag, train_names)
        Theta = model.fit(piece)
        self.save(condition_name, Theta)
        return Theta

    def estimate(self, Theta: np.ndarray, U: np.ndarray) -> np.ndarray:
        Phi = build_noncausal_design(U, self.past_order, self.future_order)
        return Phi @ Theta

    def update_with_new_data(self, Theta: np.ndarray, U_new: np.ndarray, Y_new: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Simple parameter update via convex combination between old Theta and new ridge solution on new data.
        """
        Phi_new = build_noncausal_design(U_new, self.past_order, self.future_order)
        # closed-form ridge with small regularization for stability
        G = Phi_new.T @ Phi_new
        lam = 1e-6 * np.trace(G) / G.shape[0]
        Theta_new = np.linalg.inv(G + lam * np.eye(G.shape[0])) @ Phi_new.T @ Y_new
        Theta_updated = (1 - alpha) * Theta + alpha * Theta_new
        return Theta_updated


