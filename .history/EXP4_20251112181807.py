import os
import argparse
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy.io as scio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from MIMOFIR import MIMOFIR
from models_manager import MIMOFIRManager


# Visualization defaults (can be tweaked from MATLAB script or here)
AXIS_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
TITLE_FONTSIZE = 13
LINEWIDTH = 1.8


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _condition_token(name: str) -> str:
    return name.split('_')[0]


def _coarse_prefix(name: str) -> str:
    token = _condition_token(name)
    return ''.join(ch for ch in token if ch.isalpha())


def _deduplicate(names: List[str]) -> List[str]:
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            unique.append(n)
            seen.add(n)
    return unique


def _parse_group_string(argument: str) -> List[List[str]]:
    """
    Convert a CLI string such as "SW20_01,SW20_02;RE20_01,RE20_02" to
    [['SW20_01','SW20_02'], ['RE20_01','RE20_02']].
    """
    if not argument:
        return []
    groups = []
    for chunk in argument.split(';'):
        names = [token.strip() for token in chunk.split(',') if token.strip()]
        if names:
            groups.append(names)
    return groups


def _normalize_groups(groups: Union[List[List[str]], List[str], Iterable[str]]) -> List[List[str]]:
    normalized: List[List[str]] = []
    if not groups:
        return normalized
    for entry in groups:
        if isinstance(entry, str):
            normalized.append([entry])
        elif isinstance(entry, Iterable):
            flat = [str(x).strip() for x in entry if str(x).strip()]
            if flat:
                normalized.append(flat)
        else:
            raise TypeError(f'Unsupported group entry type: {type(entry)}')
    return normalized


def _merge_by_coarse(groups: List[List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for group in groups:
        if not group:
            continue
        prefix = _coarse_prefix(group[0])
        merged.setdefault(prefix, [])
        merged[prefix].extend(group)
    for prefix, names in merged.items():
        merged[prefix] = _deduplicate(names)
    return merged


def _load_data_for_group(group: List[str],
                         past_order: int,
                         future_order: int,
                         channel_in: List[int],
                         channel_out: List[int],
                         mean_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
    loader = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, group)
    U, Y = loader._load_files()
    return U.copy(), Y.copy()


def _compute_time_axis(n_samples: int, sample_rate: float) -> np.ndarray:
    return np.arange(n_samples, dtype=np.float32) / float(sample_rate)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-8
    residual = y_true - y_pred
    rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    denom = np.sum(np.square(y_true - np.mean(y_true, axis=0, keepdims=True)), axis=0)
    r2 = 1.0 - (np.sum(np.square(residual), axis=0) / np.clip(denom, eps, None))
    mape = np.mean(np.abs(residual) / np.clip(np.abs(y_true), eps, None), axis=0)
    return {
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def _moving_average_matrix(U: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(U, ((pad, pad), (0, 0)), mode='edge')
    smoothed = np.zeros_like(U)
    for idx in range(U.shape[1]):
        smoothed[:, idx] = np.convolve(padded[:, idx], kernel, mode='valid')
    return smoothed


def simulate_full_load_inputs(U: np.ndarray,
                              scale_gain: float = 0.08,
                              bias_scale: float = 0.02) -> np.ndarray:
    """
    Increase overall load by scaling inputs and adding a bias proportional to the
    channel-wise standard deviation.
    """
    bias = bias_scale * np.std(U, axis=0, keepdims=True)
    U[:,4]
    return U * (1.0 + scale_gain) + bias


def simulate_winter_tire_inputs(U: np.ndarray,
                                smoothing_window: int = 9,
                                damping: float = 0.35) -> np.ndarray:
    """
    Winter tires tend to soften/highly damp the response; mimic this by
    blending the raw inputs with a smoothed trace.
    """
    smoothed = _moving_average_matrix(U, smoothing_window)
    return (1.0 - damping) * U + damping * smoothed


def simulate_sport_suspension_inputs(U: np.ndarray,
                                     stiffness_gain: float = 0.15) -> np.ndarray:
    """
    Sport suspension transmits more high-frequency content.
    Approximate this by injecting a scaled discrete derivative.
    """
    delta = np.diff(U, axis=0, prepend=U[:1, :])
    return U + stiffness_gain * delta


def evaluate_groups(groups: List[List[str]],
                    manager: MIMOFIRManager,
                    theta_cache: Dict[str, np.ndarray],
                    modifier: Optional[Callable[[np.ndarray], np.ndarray]],
                    sample_rate: float) -> Dict[str, np.ndarray]:
    all_true = []
    all_pred = []
    used_groups: List[str] = []
    for group in groups:
        if not group:
            continue
        prefix = _coarse_prefix(group[0])
        if prefix not in theta_cache:
            raise RuntimeError(f'No trained model for prefix "{prefix}".')
        U, Y = _load_data_for_group(group,
                                    manager.past_order,
                                    manager.future_order,
                                    manager.channel_in,
                                    manager.channel_out,
                                    manager.mean_flag)
        U_mod = modifier(U) if modifier else U
        Y_hat = manager.estimate(theta_cache[prefix], U_mod)
        all_true.append(Y)
        all_pred.append(Y_hat)
        used_groups.append('+'.join(group))
    if not all_true:
        raise RuntimeError('Empty group list provided for evaluation.')
    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    time_axis = _compute_time_axis(y_true.shape[0], sample_rate)
    metrics = _compute_metrics(y_true, y_pred)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'time': time_axis,
        'metrics': metrics,
        'groups': np.array(used_groups, dtype=object)
    }


def save_experiment_outputs(exp_name: str,
                            base_data: Dict[str, np.ndarray],
                            changed_data: Dict[str, np.ndarray],
                            fz_index: int,
                            output_dir: str) -> Dict[str, np.ndarray]:
    fz_true_base = base_data['y_true'][:, fz_index]
    fz_pred_base = base_data['y_pred'][:, fz_index]
    fz_true_mod = changed_data['y_true'][:, fz_index]
    fz_pred_mod = changed_data['y_pred'][:, fz_index]

    payload = {
        'experiment': np.array(exp_name, dtype=object),
        'time_base': base_data['time'],
        'time_modified': changed_data['time'],
        'Fz_true_base': fz_true_base,
        'Fz_pred_base': fz_pred_base,
        'Fz_true_modified': fz_true_mod,
        'Fz_pred_modified': fz_pred_mod,
        'baseline_groups': base_data['groups'],
        'modified_groups': changed_data['groups'],
        'rmse_base': base_data['metrics']['rmse'],
        'rmse_modified': changed_data['metrics']['rmse'],
        'r2_base': base_data['metrics']['r2'],
        'r2_modified': changed_data['metrics']['r2'],
        'mape_base': base_data['metrics']['mape'],
        'mape_modified': changed_data['metrics']['mape']
    }
    mat_path = os.path.join(output_dir, f'{exp_name}.mat')
    scio.savemat(mat_path, payload)
    return {
        'fz_true_base': fz_true_base,
        'fz_pred_base': fz_pred_base,
        'fz_true_mod': fz_true_mod,
        'fz_pred_mod': fz_pred_mod,
        'time_base': base_data['time'],
        'time_mod': changed_data['time'],
        'mat_path': mat_path
    }


def plot_experiment(exp_name: str,
                    data_bundle: Dict[str, np.ndarray],
                    output_dir: str) -> None:
    measured_color = '#1b9e77'
    baseline_color = '#d95f02'
    changed_color = '#7570b3'

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=False)
    fig.suptitle(exp_name.replace('_', ' ').title(), fontsize=TITLE_FONTSIZE + 1)

    axes[0].plot(data_bundle['time_base'], data_bundle['fz_true_base'], color=measured_color,
                 label='Measured $F_z$', linewidth=LINEWIDTH)
    axes[0].plot(data_bundle['time_base'], data_bundle['fz_pred_base'], color=baseline_color,
                 label='Predicted $F_z$', linewidth=LINEWIDTH)
    axes[0].set_title('No configuration change', fontsize=TITLE_FONTSIZE)
    axes[0].set_ylabel('$F_z$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].grid(alpha=0.2, linestyle='--')
    axes[0].legend(fontsize=LEGEND_FONTSIZE, frameon=False)

    axes[1].plot(data_bundle['time_mod'], data_bundle['fz_true_mod'], color=measured_color,
                 label='Measured $F_z$', linewidth=LINEWIDTH)
    axes[1].plot(data_bundle['time_mod'], data_bundle['fz_pred_mod'], color=changed_color,
                 label='Predicted $F_z$', linewidth=LINEWIDTH)
    axes[1].set_title('With configuration change', fontsize=TITLE_FONTSIZE)
    axes[1].set_xlabel('Time (s)', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel('$F_z$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].grid(alpha=0.2, linestyle='--')
    axes[1].legend(fontsize=LEGEND_FONTSIZE, frameon=False)

    for ax in axes:
        ax.tick_params(labelsize=LEGEND_FONTSIZE)

    outfile = os.path.join(output_dir, f'{exp_name}.png')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


def prepare_models(trainlist: List[List[str]], manager: MIMOFIRManager) -> Dict[str, np.ndarray]:
    merged = _merge_by_coarse(trainlist)
    if not merged:
        raise RuntimeError('Training list is empty.')
    cache: Dict[str, np.ndarray] = {}
    for prefix, names in merged.items():
        if manager.exists(prefix):
            Theta = manager.load(prefix)
        else:
            Theta = manager.train_from_files(prefix, names, piece='all')
        cache[prefix] = Theta
    return cache


def run_experiment(exp_name: str,
                   description: str,
                   manager: MIMOFIRManager,
                   theta_cache: Dict[str, np.ndarray],
                   testlist: List[List[str]],
                   changelist: List[List[str]],
                   modifier: Optional[Callable[[np.ndarray], np.ndarray]],
                   sample_rate: float,
                   fz_index: int,
                   output_dir: str) -> Dict[str, np.ndarray]:
    base_data = evaluate_groups(testlist, manager, theta_cache, modifier=None, sample_rate=sample_rate)
    changed_data = evaluate_groups(changelist, manager, theta_cache, modifier=modifier, sample_rate=sample_rate)
    bundle = save_experiment_outputs(exp_name, base_data, changed_data, fz_index, output_dir)
    plot_experiment(exp_name, bundle, output_dir)
    rmse_base = float(base_data['metrics']['rmse'][fz_index])
    rmse_mod = float(changed_data['metrics']['rmse'][fz_index])
    print(f'[{exp_name}] {description}')
    print(f'    Baseline RMSE (Fz): {rmse_base:.4f}')
    print(f'    Changed  RMSE (Fz): {rmse_mod:.4f}')
    print(f'    Data saved to: {bundle["mat_path"]}')
    print(f'    Figure saved to: {os.path.join(output_dir, f"{exp_name}.png")}')
    return bundle


def run_loading_conditions(manager: MIMOFIRManager,
                           theta_cache: Dict[str, np.ndarray],
                           testlist: List[List[str]],
                           changelist: List[List[str]],
                           sample_rate: float,
                           fz_index: int,
                           output_dir: str) -> Dict[str, np.ndarray]:
    modifier = lambda U: simulate_full_load_inputs(U)  # noqa: E731
    return run_experiment('loading_conditions',
                          'Baseline vs full loading',
                          manager,
                          theta_cache,
                          testlist,
                          changelist,
                          modifier,
                          sample_rate,
                          fz_index,
                          output_dir)


def run_tire_changes(manager: MIMOFIRManager,
                     theta_cache: Dict[str, np.ndarray],
                     testlist: List[List[str]],
                     changelist: List[List[str]],
                     sample_rate: float,
                     fz_index: int,
                     output_dir: str) -> Dict[str, np.ndarray]:
    modifier = lambda U: simulate_winter_tire_inputs(U)  # noqa: E731
    return run_experiment('tire_changes',
                          'OE tires vs winter tires',
                          manager,
                          theta_cache,
                          testlist,
                          changelist,
                          modifier,
                          sample_rate,
                          fz_index,
                          output_dir)


def run_suspension_mods(manager: MIMOFIRManager,
                        theta_cache: Dict[str, np.ndarray],
                        testlist: List[List[str]],
                        changelist: List[List[str]],
                        sample_rate: float,
                        fz_index: int,
                        output_dir: str) -> Dict[str, np.ndarray]:
    modifier = lambda U: simulate_sport_suspension_inputs(U)  # noqa: E731
    return run_experiment('suspension_mods',
                          'Standard vs sport suspension',
                          manager,
                          theta_cache,
                          testlist,
                          changelist,
                          modifier,
                          sample_rate,
                          fz_index,
                          output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EXP4 configuration-sensitivity experiments.')
    parser.add_argument('--trainlist', type=str, default='SW20_01,SW20_02',
                        help='Semicolon-separated groups, comma-separated names per group.')
    parser.add_argument('--testlist', type=str, default='SW20_02',
                        help='Semicolon-separated groups used for baseline inference.')
    parser.add_argument('--changelist', type=str, default='SW20_03',
                        help='Semicolon-separated groups used for configuration-change tests.')
    parser.add_argument('--base-name', type=str, default='',
                        help='Optional shorthand to auto-build lists as Base_01/Base_02/Base_03.')
    parser.add_argument('--scenario-name', type=str, default='',
                        help='Optional name appended to output directory (defaults to first condition token).')
    parser.add_argument('--channel-in', dest='channel_in', nargs='+', type=int,
                        default=[69, 78, 83, 84, 91, 94, 95, 96],
                        help='Input channel indices.')
    parser.add_argument('--channel-out', dest='channel_out', nargs='+', type=int,
                        default=[32, 33, 34, 35, 36, 37],
                        help='Output channel indices.')
    parser.add_argument('--past-order', type=int, default=50, help='Past FIR order.')
    parser.add_argument('--future-order', type=int, default=50, help='Future FIR order.')
    parser.add_argument('--sample-rate', type=float, default=500.0, help='Sampling rate in Hz.')
    parser.add_argument('--fz-index', type=int, default=2,
                        help='Column index of Fz within channel_out (0-based).')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory for cached FIR models.')
    parser.add_argument('--output-dir', type=str, default=os.path.join('result', 'EXP4'),
                        help='Directory to store experiment data and figures.')
    parser.add_argument('--no-mean', dest='mean_flag', action='store_false',
                        help='Disable mean-centering before training/evaluation.')
    parser.set_defaults(mean_flag=True)
    return parser.parse_args()


def main() -> None:
    plt.rcParams['font.family'] = 'Times New Roman'
    args = parse_args()

    if args.base_name:
        base = args.base_name.strip()
        trainlist = [[f'{base}_01', f'{base}_02']]
        testlist = [[f'{base}_02']]
        changelist = [[f'{base}_03']]
    else:
        trainlist = _parse_group_string(args.trainlist)
        testlist = _parse_group_string(args.testlist)
        changelist = _parse_group_string(args.changelist)

    trainlist = _normalize_groups(trainlist)
    testlist = _normalize_groups(testlist)
    changelist = _normalize_groups(changelist)

    if not trainlist or not testlist or not changelist:
        raise RuntimeError('Please provide non-empty trainlist, testlist, and changelist.')

    manager = MIMOFIRManager(args.past_order,
                             args.future_order,
                             args.channel_in,
                             args.channel_out,
                             args.mean_flag,
                             models_dir=args.models_dir)

    theta_cache = prepare_models(trainlist, manager)

    scenario_name = args.scenario_name.strip()
    if not scenario_name:
        scenario_name = args.base_name.strip() if args.base_name else _condition_token(trainlist[0][0])

    output_dir = os.path.join(args.output_dir, scenario_name)
    _ensure_dir(output_dir)

    run_loading_conditions(manager, theta_cache, testlist, changelist, args.sample_rate, args.fz_index, output_dir)
    run_tire_changes(manager, theta_cache, testlist, changelist, args.sample_rate, args.fz_index, output_dir)
    run_suspension_mods(manager, theta_cache, testlist, changelist, args.sample_rate, args.fz_index, output_dir)
    print('EXP4 experiments completed.')


if __name__ == '__main__':
    main()
