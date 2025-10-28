import os
import numpy as np
import scipy.io as scio
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from feature_library import build_feature_library, compute_feature_for_name, extract_condition_prefix
from models_manager import MIMOFIRManager
from MIMOFIR import MIMOFIR
from recognizers import get_recognizer
import config


def read_io_for_names(names: List[str], channel_in: List[int], channel_out: List[int], mean_flag: bool) -> tuple:
    model = MIMOFIR(past_order=1, future_order=0, channel_in=channel_in, channel_out=channel_out, mean_flag=mean_flag, train_names=names)
    U, Y = model._load_files()
    return U, Y


def pipeline(trainlist: List[List[str]],
             testlist: List[List[str]],
             channel_in: List[int],
             channel_out: List[int],
             past_order: int = 50,
             future_order: int = 50,
             mean_flag: bool = True,
             max_unknown: int = 3,
             feature_dir: str = 'feature',
             models_dir: str = 'models',
             results_dir: str = os.path.join('result', 'pipeline')) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # 1) Build feature library for known conditions (persist features for analysis/EVM)
    build_feature_library(trainlist, feature_dir=feature_dir)

    # 2) Build recognizer based on config and fit
    recognizer = get_recognizer(
        strategy=config.RECOGNITION_STRATEGY,
        evm_tail_frac=getattr(config, 'EVM_TAIL_FRAC', 0.5),
        evm_reject_threshold=getattr(config, 'EVM_REJECT_THRESHOLD', 0.5),
        hmm_n_states=getattr(config, 'HMM_N_STATES', 3),
        hmm_cov_reg=getattr(config, 'HMM_COV_REG', 1e-6),
        hmm_n_iter=getattr(config, 'HMM_N_ITER', 15),
        hmm_glrt_quantile=getattr(config, 'HMM_GLRT_QUANTILE', 0.1),
        random_state=getattr(config, 'RANDOM_STATE', 0),
        hmm_feature_mode=getattr(config, 'HMM_FEATURE_MODE', 'stft'),
        hmm_num_bands=getattr(config, 'HMM_NUM_BANDS', 4),
        hmm_raw_win=getattr(config, 'HMM_RAW_WIN', 32),
        hmm_raw_hop=getattr(config, 'HMM_RAW_HOP', 16),
        hmm_standardize=getattr(config, 'HMM_STANDARDIZE', True),
        hmm_restarts=getattr(config, 'HMM_RESTARTS', 1),
        hmm_self_transition=getattr(config, 'HMM_SELF_TRANSITION', 0.9),
        hmm_use_avg_ll=getattr(config, 'HMM_USE_AVG_LL', True),
    )
    recognizer.fit_groups(trainlist)

    # 3) Prepare MIMOFIR manager
    manager = MIMOFIRManager(past_order, future_order, channel_in, channel_out, mean_flag, models_dir=models_dir)

    # 4) Ensure models exist for known classes
    for group in trainlist:
        prefix = extract_condition_prefix(group)
        if not manager.exists(prefix):
            manager.train_from_files(prefix, group, piece='all')

    # unknown pool management
    unknown_labels: List[str] = []

    # combined storage for final plot
    combined_true: List[np.ndarray] = []
    combined_pred: List[np.ndarray] = []
    combined_labels: List[str] = []
    segment_lengths: List[int] = []

    # 5) Sequentially process test conditions
    for test_names in testlist:
        test_name = test_names[0]
        print('-------------------Processing Test Condition-------------------')
        print(f'Reading test condition: {test_name}')

        # Recognize condition via configured strategy
        cls_hat, prob, scores = recognizer.predict_on_name(test_name)

        # Log recognition scores
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_items = ', '.join([f"{k}:{v:.3f}" for k, v in sorted_scores[:5]])
        print(f'Recognizer({config.RECOGNITION_STRATEGY}) predicted (top-5): {top_items}')
        print(f'Recognizer decision: {cls_hat} (p={prob:.3f})')

        # Read input/output for this test
        U_test, Y_test = read_io_for_names(test_names, channel_in, channel_out, mean_flag)

        if cls_hat == 'Unknown':
            # choose most similar known class as fallback for estimation
            if len(scores) == 0:
                # if completely empty, skip
                continue
            fallback = max(scores, key=scores.get)
            print(f'Unknown condition detected. Using fallback known class for estimation: {fallback}')
            Theta = manager.load(fallback)
            if Theta is None:
                Theta = manager.train_from_files(fallback, [fallback + '_01', fallback + '_02'], piece='all')
            Y_pred = manager.estimate(Theta, U_test)

            # manage unknown labels pool (max 3)
            new_unknown = None
            # pick a new label name
            for k in range(1, max_unknown + 1):
                label = f'Unknown{k}'
                if label not in unknown_labels:
                    new_unknown = label
                    break
            if new_unknown is None:
                # replace the least probable existing unknown class: here we just keep first 3, ignore others
                new_unknown = unknown_labels[-1]
            if new_unknown not in unknown_labels:
                unknown_labels.append(new_unknown)

            # Train model for unknown on-the-fly
            if not manager.exists(new_unknown):
                # initial training from current test sample only
                # Solve Theta directly from this sample (ridge on this single dataset)
                model_tmp = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, test_names)
                Theta_unknown = model_tmp.fit(piece='all')
                manager.save(new_unknown, Theta_unknown)
            print(f'Assigned label: {new_unknown}. Unknown pool: {unknown_labels}')

            # Save prediction
            out_dir = os.path.join(results_dir, test_name)
            os.makedirs(out_dir, exist_ok=True)
            scio.savemat(os.path.join(out_dir, 'prediction.mat'), {'Y_true': Y_test, 'Y_pred': Y_pred, 'class': new_unknown})

            # Metrics and per-test plot
            _save_test_metrics_and_plot(Y_test, Y_pred, channel_out, new_unknown, out_dir, test_name)
            combined_true.append(Y_test)
            combined_pred.append(Y_pred)
            combined_labels.append(new_unknown)
            segment_lengths.append(Y_test.shape[0])
        else:
            # known class, ensure model and estimate
            Theta = manager.load(cls_hat)
            if Theta is None:
                # train from existing known data group
                for group in trainlist:
                    if extract_condition_prefix(group) == cls_hat:
                        Theta = manager.train_from_files(cls_hat, group, piece='all')
                        break
            Y_pred = manager.estimate(Theta, U_test)

            # update model with new data
            Theta_updated = manager.update_with_new_data(Theta, U_test, Y_test, alpha=0.2)
            manager.save(cls_hat, Theta_updated)

            # Save prediction
            out_dir = os.path.join(results_dir, test_name)
            os.makedirs(out_dir, exist_ok=True)
            scio.savemat(os.path.join(out_dir, 'prediction.mat'), {'Y_true': Y_test, 'Y_pred': Y_pred, 'class': cls_hat})

            # Metrics and per-test plot
            _save_test_metrics_and_plot(Y_test, Y_pred, channel_out, cls_hat, out_dir, test_name)
            combined_true.append(Y_test)
            combined_pred.append(Y_pred)
            combined_labels.append(cls_hat)
            segment_lengths.append(Y_test.shape[0])

    # Combined plot across all tests
    if len(combined_true) > 0:
        _plot_combined_ieee(combined_true, combined_pred, combined_labels, segment_lengths, channel_out, results_dir)


def _ensure_ieee_style() -> None:
    try:
        plt.style.use(['science', 'ieee'])
    except Exception:
        plt.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'


def _save_test_metrics_and_plot(Y_true: np.ndarray,
                                Y_pred: np.ndarray,
                                channel_out: List[int],
                                label_str: str,
                                out_dir: str,
                                test_name: str) -> None:
    rmse = float(np.sqrt(mean_squared_error(Y_true, Y_pred)))
    r2 = float(r2_score(Y_true, Y_pred))
    mape = float(mean_absolute_percentage_error(Y_true, Y_pred))
    print(f'Metrics for {test_name} [{label_str}] -> RMSE: {rmse:.6f}, R2: {r2:.6f}, MAPE: {mape:.6f}')
    with open(os.path.join(out_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{test_name} [{label_str}]\n')
        f.write(f'RMSE: {rmse:.6f}\nR2: {r2:.6f}\nMAPE: {mape:.6f}\n')

    _ensure_ieee_style()
    n_outputs = Y_true.shape[1]
    n_cols = 2 if n_outputs > 1 else 1
    n_rows = int(np.ceil(n_outputs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 1.8 * n_rows), constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    t = np.arange(Y_true.shape[0]) / 500.0
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
    total_axes = n_rows * n_cols
    for j in range(n_outputs, total_axes):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r, c])
    fig.suptitle(f'{test_name} - {label_str}', fontsize=12)
    png_path = os.path.abspath(os.path.join(out_dir, f'{test_name}_ieee.png'))
    pdf_path = os.path.abspath(os.path.join(out_dir, f'{test_name}_ieee.pdf'))
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    try:
        with open(pdf_path, 'wb') as fh:
            fig.savefig(fh, format='pdf', dpi=600, bbox_inches='tight')
    except Exception as e:
        print(f'Warning: failed to save PDF to {pdf_path}: {e}')
    plt.close(fig)


def _plot_combined_ieee(list_Y_true: List[np.ndarray],
                        list_Y_pred: List[np.ndarray],
                        list_labels: List[str],
                        segment_lengths: List[int],
                        channel_out: List[int],
                        results_dir: str) -> None:
    _ensure_ieee_style()
    Y_true = np.vstack(list_Y_true)
    Y_pred = np.vstack(list_Y_pred)
    n_outputs = Y_true.shape[1]
    n_cols = 2 if n_outputs > 1 else 1
    n_rows = int(np.ceil(n_outputs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 2.0 * n_rows), constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    # build time axis and boundaries
    total_len = Y_true.shape[0]
    t = np.arange(total_len) / 500.0
    boundaries = np.cumsum(segment_lengths)[:-1]
    for i in range(n_outputs):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        ax.plot(t, Y_true[:, i], label='Measured', color='#2299f0', linewidth=1.2)
        ax.plot(t, Y_pred[:, i], label='Estimated', color='#72b607', linestyle='--', linewidth=1.2)
        for b in boundaries:
            ax.axvline(x=b / 500.0, color='k', linestyle='--', linewidth=0.8)
        # annotate labels roughly at mid of segment on top
        start = 0
        for seg_len, lbl in zip(segment_lengths, list_labels):
            mid = start + seg_len // 2
            ax.text(mid / 500.0, ax.get_ylim()[1], lbl, fontsize=9, ha='center', va='bottom')
            start += seg_len
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Ch {channel_out[i]}')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=10)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_linewidth(1.0)
        if i == 0:
            ax.legend(frameon=True, edgecolor='black', fontsize=9, fancybox=False, ncol=1, loc='upper right')
    total_axes = n_rows * n_cols
    for j in range(n_outputs, total_axes):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r, c])
    fig.suptitle('Combined Test Results', fontsize=12)
    png_path = os.path.abspath(os.path.join(results_dir, 'combined_ieee.png'))
    pdf_path = os.path.abspath(os.path.join(results_dir, 'combined_ieee.pdf'))
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    try:
        with open(pdf_path, 'wb') as fh:
            fig.savefig(fh, format='pdf', dpi=600, bbox_inches='tight')
    except Exception as e:
        print(f'Warning: failed to save PDF to {pdf_path}: {e}')
    plt.close(fig)


if __name__ == '__main__':
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
        ['SW20_03'], ['SW25_03'], ['SW35_03'],
        ['RE20_03'], ['RE30_03'], ['RE40_03'],
        ['CJ30_03'], ['CJ40_03'], ['CJ50_03'],
    ]

    pipeline(trainlist, testlist, channel_in, channel_out, past_order, future_order, mean_flag)


