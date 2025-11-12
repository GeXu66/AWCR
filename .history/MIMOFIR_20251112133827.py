import numpy as np
import scipy.io as scio
import os
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import json


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
        condition = names[0].split('_')[0]
        out_dir = os.path.join('result', 'MIMOFIR', condition)
        os.makedirs(out_dir, exist_ok=True)

        model = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, names)
        # Load training data explicitly for metrics and plotting
        U_train_all, Y_train_all = model._load_files()
        a, b = _validate_slice(piece, U_train_all.shape[0])
        U_train = U_train_all[a:b + 1, :]
        Y_train = Y_train_all[a:b + 1, :]

        Phi_train = build_noncausal_design(U_train, past_order, future_order)
        Theta = _ridge_regression(Phi_train, Y_train)
        model.Theta = Theta
        thetas.append(Theta)

        # Training predictions and metrics
        Y_pred_train = Phi_train @ Theta
        rmse = float(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
        r2 = float(r2_score(Y_train, Y_pred_train))
        # MAPE with numerical stability
        eps = 1e-8
        mape = float(mean_absolute_percentage_error(Y_train, Y_pred_train))

        # Log per condition
        print('-------------------Training Log-------------------')
        print(f'Condition: {condition} | Train files: {names}')
        print(f'Design matrix shape: {Phi_train.shape}, Theta shape: {Theta.shape}')
        print(f'RMSE: {rmse:.6f} | R2: {r2:.6f} | MAPE: {mape:.6f}')

        # Save predictions and metrics
        mat_path = os.path.join(out_dir, f'{condition}_train_pred.mat')
        scio.savemat(mat_path, {
            'Y_true': Y_train,
            'Y_pred': Y_pred_train,
            'Theta': Theta,
            'channel_in': np.array(channel_in).reshape(-1, 1),
            'channel_out': np.array(channel_out).reshape(-1, 1),
            'past_order': past_order,
            'future_order': future_order,
        })

        metrics = {'RMSE': rmse, 'R2': r2, 'MAPE': mape,
                   'n_samples': int(Y_train.shape[0]), 'n_outputs': int(Y_train.shape[1])}
        with open(os.path.join(out_dir, f'{condition}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Plot training results in IEEE style
        _plot_training_ieee(Y_train, Y_pred_train, channel_out, condition, out_dir)
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


def _plot_training_ieee(Y_true: np.ndarray,
                        Y_pred: np.ndarray,
                        channel_out: List[int],
                        condition: str,
                        out_dir: str) -> None:
    # Try to use SciencePlots with IEEE
    try:
        plt.style.use(['science', 'ieee'])
    except Exception:
        # Fallback to default style with Times New Roman
        plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'

    n_outputs = Y_true.shape[1]
    # Arrange subplots in up to 3x2 for 6 outputs; generalize grid
    n_cols = 2 if n_outputs > 1 else 1
    n_rows = int(np.ceil(n_outputs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 1.8 * n_rows), constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])

    t = np.arange(Y_true.shape[0]) / 500.0  # assuming 500 Hz if applicable
    for i in range(n_outputs):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        ax.plot(t, Y_true[:, i], label='Measured', color='#2299f0', linewidth=1.5)
        ax.plot(t, Y_pred[:, i], label='Estimated', color='#72b607', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Ch {channel_out[i]}')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=10)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_linewidth(1.0)
        if i == 0:
            ax.legend(frameon=True, edgecolor='black', fontsize=9, fancybox=False, ncol=1, loc='upper right')

    # Hide unused subplots
    total_axes = n_rows * n_cols
    for j in range(n_outputs, total_axes):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r, c])

    fig.suptitle(f'Training Results - {condition}', fontsize=12)
    png_path = os.path.join(out_dir, f'{condition}_train_ieee.png')
    pdf_path = os.path.join(out_dir, f'{condition}_train_ieee.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


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


