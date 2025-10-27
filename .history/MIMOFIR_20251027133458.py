import numpy as np
import scipy.io as scio
import os
from typing import List, Tuple, Optional, Union


def _ridge_regression(K: np.ndarray, Y: np.ndarray, cond_limit: float = 1e9) -> np.ndarray:
    """
    Solve Theta = argmin ||K Theta - Y||^2 + lambda ||Theta||^2 with data-driven lambda
    that keeps condition number below cond_limit (same logic as in FIR.ridge_reg).
    K: (N, P)
    Y: (N, M)
    Returns Theta: (P, M)
    """
    cond_num = np.linalg.cond(K)
    G = K.T @ K
    eigvals = np.linalg.eigvals(G)
    max_eig = float(np.real(np.max(eigvals))) if eigvals.size > 0 else 0.0
    min_eig = float(np.real(np.min(eigvals))) if eigvals.size > 0 else 0.0
    if cond_num > cond_limit and max_eig > 0:
        lam = (max_eig - min_eig * cond_limit) / (cond_limit - 1)
    else:
        lam = 0.0
    inv_G = np.linalg.inv(G + lam * np.eye(G.shape[0]))
    Theta = inv_G @ K.T @ Y
    return Theta


def _validate_slice(piece: Union[str, Tuple[int, int]], length: int) -> Tuple[int, int]:
    if isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[0], int) and isinstance(piece[1], int):
        a, b = piece
        if 0 <= a <= b < length:
            return a, b
        raise ValueError("piece tuple out of range")
    if piece == 'all':
        return 0, length - 1
    raise ValueError("piece must be ('all') or (start, end)")


def _concat_trials(trains: List[np.ndarray], targets: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(trains) == 0:
        raise ValueError("empty train list")
    U = trains[0]
    Y = targets[0]
    for i in range(1, len(trains)):
        U = np.vstack((U, trains[i]))
        Y = np.vstack((Y, targets[i]))
    return U, Y


def _standardize(X: np.ndarray, center: bool) -> np.ndarray:
    if not center:
        return X
    return X - np.mean(X, axis=0, keepdims=True)


def build_noncausal_design(U: np.ndarray, past_order: int, future_order: int) -> np.ndarray:
    """
    Build noncausal FIR design matrix for multi-input U (T, ni).
    Row t contains [u(t), u(t-1),...,u(t-past_order+1), u(t+1),...,u(t+future_order)] stacked for all inputs.
    Edge handling via padding first/last rows.
    Returns Phi: (T, ni * (past_order + future_order + 1))
    """
    T, ni = U.shape
    total_order = past_order + future_order + 1
    Phi = np.zeros((T, ni * total_order))

    # Pad for past with first row, for future with last row
    if past_order > 0:
        past_pad = np.tile(U[0, :], (past_order, 1))
    else:
        past_pad = np.zeros((0, ni))
    if future_order > 0:
        future_pad = np.tile(U[-1, :], (future_order, 1))
    else:
        future_pad = np.zeros((0, ni))

    U_ext = np.vstack((past_pad, U, future_pad))  # shape: (T + past + future, ni)

    # For each time t in 0..T-1, take window centered at t+past_order
    for t in range(T):
        start = t
        end = t + total_order
        window = U_ext[start:end, :]  # (total_order, ni)
        # Order: current, past lags, future leads
        # window rows are [u(t-past), ..., u(t), ..., u(t+future)]
        # We want [u(t), u(t-1),...,u(t-past+1), u(t+1),...,u(t+future)]
        idx_current = past_order
        parts = [window[idx_current, :][None, :]]
        if past_order > 0:
            parts.append(np.flipud(window[:idx_current, :]))
        if future_order > 0:
            parts.append(window[idx_current + 1:, :])
        stack = np.vstack(parts) if len(parts) > 1 else parts[0]
        Phi[t, :] = stack.T.reshape(-1)
    return Phi


class MIMOFIR:
    def __init__(self,
                 past_order: int,
                 future_order: int,
                 channel_in: List[int],
                 channel_out: List[int],
                 mean_flag: bool,
                 train_names: Union[str, List[str]]):
        self.past_order = int(past_order)
        self.future_order = int(future_order)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.mean_flag = mean_flag
        self.ni = len(channel_in)
        self.no = len(channel_out)
        if isinstance(train_names, list):
            self.train_files = ['../Data/matdata/' + n + '.mat' for n in train_names]
        else:
            self.train_files = ['../Data/matdata/' + train_names + '.mat']
        self.Theta: Optional[np.ndarray] = None  # (P, no)

    def _load_files(self) -> Tuple[np.ndarray, np.ndarray]:
        trains = []
        targets = []
        for file in self.train_files:
            key = os.path.basename(file).split('.')[-2]
            data = scio.loadmat(file)[key]
            trains.append(data[:, self.channel_in])
            targets.append(data[:, self.channel_out])
        U, Y = _concat_trials(trains, targets)
        if self.mean_flag:
            U = _standardize(U, True)
            Y = _standardize(Y, True)
        return U, Y

    def fit(self, piece: Union[str, Tuple[int, int]] = 'all') -> np.ndarray:
        U, Y = self._load_files()
        a, b = _validate_slice(piece, U.shape[0])
        U = U[a:b + 1, :]
        Y = Y[a:b + 1, :]
        Phi = build_noncausal_design(U, self.past_order, self.future_order)
        self.Theta = _ridge_regression(Phi, Y)
        return self.Theta

    def transform_inputs(self, U: np.ndarray) -> np.ndarray:
        return build_noncausal_design(U, self.past_order, self.future_order)

    def predict_from_inputs(self, U: np.ndarray) -> np.ndarray:
        if self.Theta is None:
            raise RuntimeError("Model not fitted.")
        Phi = self.transform_inputs(U)
        return Phi @ self.Theta


def fit_per_condition(past_order: int,
                      future_order: int,
                      channel_in: List[int],
                      channel_out: List[int],
                      mean_flag: bool,
                      train_groups: List[List[str]],
                      piece: Union[str, Tuple[int, int]] = 'all') -> List[np.ndarray]:
    """
    Train a parameter matrix per condition group in train_groups.
    Returns list of Theta matrices, each of shape (P, no).
    """
    thetas = []
    for names in train_groups:
        model = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, names)
        Theta = model.fit(piece)
        thetas.append(Theta)
    return thetas


def evaluate_on_tests(thetas: List[np.ndarray],
                      past_order: int,
                      future_order: int,
                      channel_in: List[int],
                      channel_out: List[int],
                      mean_flag: bool,
                      train_groups: List[List[str]],
                      test_groups: List[List[str]],
                      piece: Union[str, Tuple[int, int]] = 'all') -> List[dict]:
    """
    Using one Theta per training condition group, evaluate on its matching test group.
    Returns list of dicts with predictions and ground truth.
    """
    results = []
    for Theta, train_names, test_names in zip(thetas, train_groups, test_groups):
        # Build a helper model to get design on test
        m_train = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, train_names)
        m_test = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, test_names)
        # Load test data
        U_test, Y_test = m_test._load_files()
        a, b = _validate_slice(piece, U_test.shape[0])
        U_test = U_test[a:b + 1, :]
        Y_test = Y_test[a:b + 1, :]
        Phi_test = m_train.transform_inputs(U_test)
        Y_pred = Phi_test @ Theta
        results.append({
            'train': train_names,
            'test': test_names,
            'Y_true': Y_test,
            'Y_pred': Y_pred,
        })
    return results


if __name__ == '__main__':
    # Default configuration per user request
    channel_in = [69, 78, 83, 84, 91, 94, 95, 96]
    channel_out = [32, 33, 34, 35, 36, 37]
    past_order = 50
    future_order = 50
    mean_flag = True

    trainlist = [
        ['SW20_01', 'SW20_02'], ['SW25_01', 'SW25_02'], ['SW35_01', 'SW35_02'],
        ['RE20_01', 'RE20_02'], ['RE30_01', 'RE30_02'], ['RE40_01', 'RE40_02'],
        ['CJ30_01', 'CJ30_02'], ['CJ40_01', 'CJ40_02'], ['CJ50_01', 'CJ50_02'],
    ]
    testlist = [
        ['SW20_03'], ['SW25_03'], ['SW35_03'], ['RE20_03'], ['RE30_03'], ['RE40_03'],
        ['CJ30_03'], ['CJ40_03'], ['CJ50_03'],
    ]

    # Train and evaluate
    thetas = fit_per_condition(past_order, future_order, channel_in, channel_out, mean_flag, trainlist, piece='all')
    results = evaluate_on_tests(thetas, past_order, future_order, channel_in, channel_out, mean_flag, trainlist, testlist, piece='all')

    # Print simple diagnostics
    for res in results:
        Y_true = res['Y_true']
        Y_pred = res['Y_pred']
        mse = np.mean((Y_true - Y_pred) ** 2)
        print(f"Train {res['train']} -> Test {res['test']} | MSE: {mse:.4f} | shape pred: {Y_pred.shape}")


