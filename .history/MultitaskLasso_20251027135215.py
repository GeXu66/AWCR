import numpy as np
import scipy.io as scio
import os
from typing import List, Tuple, Optional, Union
from sklearn.linear_model import MultiTaskLasso


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


def build_noncausal_design_rows(U: np.ndarray,
                                past_order: int,
                                future_order: int,
                                row_indices: np.ndarray,
                                dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Build noncausal design only for selected time indices to reduce memory.
    Returns Phi_sel: (#rows, ni * (past+future+1))
    """
    T, ni = U.shape
    total_order = past_order + future_order + 1
    Phi_sel = np.zeros((len(row_indices), ni * total_order), dtype=dtype)

    # padding
    past_pad = np.tile(U[0, :], (past_order, 1)) if past_order > 0 else np.zeros((0, ni), dtype=U.dtype)
    future_pad = np.tile(U[-1, :], (future_order, 1)) if future_order > 0 else np.zeros((0, ni), dtype=U.dtype)
    U_ext = np.vstack((past_pad, U, future_pad))

    for j, t in enumerate(row_indices):
        start = t
        end = t + total_order
        window = U_ext[start:end, :]
        idx_current = past_order
        parts = [window[idx_current, :][None, :]]
        if past_order > 0:
            parts.append(np.flipud(window[:idx_current, :]))
        if future_order > 0:
            parts.append(window[idx_current + 1:, :])
        stack = np.vstack(parts) if len(parts) > 1 else parts[0]
        Phi_sel[j, :] = stack.T.reshape(-1).astype(dtype, copy=False)
    return Phi_sel


def stability_selection_scores_from_U(U: np.ndarray,
                                      Y: np.ndarray,
                                      past_order: int,
                                      future_order: int,
                                      n_bootstrap: int = 100,
                                      subsample_frac: float = 0.7,
                                      max_samples_per_bootstrap: int = 5000,
                                      alphas: Optional[List[float]] = None,
                                      random_state: int = 42) -> np.ndarray:
    """
    Compute stability scores without building full design matrix.
    On each bootstrap, select a subset of time indices, build the design only for those rows.
    """
    rng = np.random.RandomState(random_state)
    n_samples = U.shape[0]
    ni = U.shape[1]
    total_order = past_order + future_order + 1
    n_features = ni * total_order
    if alphas is None:
        alphas = np.logspace(-4, 0, 6)
    counts = np.zeros(n_features, dtype=float)

    for _ in range(n_bootstrap):
        k = min(int(n_samples * subsample_frac), max_samples_per_bootstrap)
        idx = rng.choice(n_samples, size=k, replace=False)

        # Build design for selected rows only (float32 to reduce memory)
        Xb = build_noncausal_design_rows(U, past_order, future_order, idx, dtype=np.float32)
        Yb = Y[idx, :].astype(np.float32, copy=False)

        # Standardize per subsample (zero-mean)
        Xb -= np.mean(Xb, axis=0, keepdims=True)
        Yb -= np.mean(Yb, axis=0, keepdims=True)

        selected = np.zeros(n_features, dtype=bool)
        for alpha in alphas:
            mtl = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=3000, selection='cyclic')
            mtl.fit(Xb, Yb)
            sel_alpha = np.any(np.abs(mtl.coef_) > 1e-12, axis=0)
            selected = np.logical_or(selected, sel_alpha)
        counts += selected.astype(float)
    scores = counts / n_bootstrap
    return scores


def run_multitask_lasso_selection(past_order: int,
                                  future_order: int,
                                  channel_in: List[int],
                                  channel_out: List[int],
                                  mean_flag: bool,
                                  train_names: Union[str, List[str]],
                                  piece: Union[str, Tuple[int, int]] = 'all',
                                  n_bootstrap: int = 100,
                                  subsample_frac: float = 0.7,
                                  alphas: Optional[List[float]] = None,
                                  random_state: int = 42) -> dict:
    # load data
    if isinstance(train_names, list):
        files = ['../Data/matdata/' + n + '.mat' for n in train_names]
    else:
        files = ['../Data/matdata/' + train_names + '.mat']
    trains, targets = [], []
    for f in files:
        key = os.path.basename(f).split('.')[-2]
        data = scio.loadmat(f)[key]
        trains.append(data[:, channel_in])
        targets.append(data[:, channel_out])
    U, Y = _concat_trials(trains, targets)
    a, b = _validate_slice(piece, U.shape[0])
    U = U[a:b + 1, :]
    Y = Y[a:b + 1, :]
    if mean_flag:
        U = _standardize(U, True)
        Y = _standardize(Y, True)

    # compute stability scores at FEATURE-GROUP level (per input channel across all time-shifts)
    # Build designs only for sampled rows to avoid huge memory usage
    scores_feature = stability_selection_scores_from_U(U, Y,
                                                       past_order=past_order,
                                                       future_order=future_order,
                                                       n_bootstrap=n_bootstrap,
                                                       subsample_frac=subsample_frac,
                                                       max_samples_per_bootstrap=5000,
                                                       alphas=alphas,
                                                       random_state=random_state)
    total_order = past_order + future_order + 1
    ni = len(channel_in)
    # reshape to (ni, total_order)
    scores_by_channel = scores_feature.reshape(ni, total_order)
    # aggregate score per input channel
    channel_scores = scores_by_channel.max(axis=1)

    ranked = np.argsort(-channel_scores)  # descending
    return {
        'channel_scores': channel_scores,
        'ranked_channel_indices': ranked,
        'ranked_channels': [channel_in[i] for i in ranked],
        'scores_by_channel_shift': scores_by_channel,
    }


if __name__ == '__main__':
    # Configuration per user request
    channel_in = [list(range(0, 32)), list(range(56, 139))]
    channel_in = [x for sublist in channel_in for x in sublist]
    remove_list = [57, 65, 76, 79]
    channel_in = [x for x in channel_in if x not in remove_list]
    channel_out = [32, 33, 34, 35, 36, 37]
    past_order = 50
    future_order = 50
    mean_flag = True

    # Use the same training conditions as in MIMOFIR for example
    trainlist = [
        ['SW20_01', 'SW20_02'], ['SW25_01', 'SW25_02'], ['SW35_01', 'SW35_02'],
        ['RE20_01', 'RE20_02'], ['RE30_01', 'RE30_02'], ['RE40_01', 'RE40_02'],
        ['CJ30_01', 'CJ30_02'], ['CJ40_01', 'CJ40_02'], ['CJ50_01', 'CJ50_02'],
    ]

    # Run on the concatenation of all training conditions for robust selection
    flat_train = [x for group in trainlist for x in group]
    res = run_multitask_lasso_selection(
        past_order=past_order,
        future_order=future_order,
        channel_in=channel_in,
        channel_out=channel_out,
        mean_flag=mean_flag,
        train_names=flat_train,
        piece='all',
        n_bootstrap=60,
        subsample_frac=0.7,
        alphas=None,
        random_state=42,
    )
    print('Ranked channels (by stability score):', res['ranked_channels'][:20])
    print('Top-10 scores:', res['channel_scores'][res['ranked_channel_indices'][:10]])


