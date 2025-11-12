import os
import re
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


def _remove_models(models_dir: str) -> None:
    # remove all mat files in models_dir
    for fn in os.listdir(models_dir):
        if fn.endswith('.mat'):
            os.remove(os.path.join(models_dir, fn))


def pipeline(trainlist: List[List[str]],
             testlist: List[List[str]],
             channel_in: List[int],
             channel_out: List[int],
             past_order: int = 50,
             future_order: int = 50,
             mean_flag: bool = True,
             max_unknown: int = 3,
             update_model: bool = True,
             feature_dir: str = 'feature',
             models_dir: str = 'models',
             results_dir: str = os.path.join('result', 'pipeline')) -> None:
    # Always purge unknown models before running a new pipeline
    _remove_models(models_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Merge training groups by coarse prefix (e.g., SW20/SW25/SW35 -> SW)
    coarse_to_names: Dict[str, List[str]] = {}
    for group in trainlist:
        prefix = extract_condition_prefix(group)
        if prefix not in coarse_to_names:
            coarse_to_names[prefix] = []
        coarse_to_names[prefix].extend(group)
    merged_trainlist: List[List[str]] = list(coarse_to_names.values())

    # 1) Build feature library for known conditions (persist features for analysis/EVM)
    build_feature_library(merged_trainlist, feature_dir=feature_dir)

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
    )
    recognizer.fit_groups(merged_trainlist)

    # 3) Prepare MIMOFIR manager
    manager = MIMOFIRManager(past_order, future_order, channel_in, channel_out, mean_flag, models_dir=models_dir)

    # 4) Ensure models exist for known classes (merged)
    for group in merged_trainlist:
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

            # manage unknown labels pool (max 3) and select the assigned unknown label
            new_unknown = None
            for k in range(1, max_unknown + 1):
                label = f'Unknown{k}'
                if label not in unknown_labels:
                    new_unknown = label
                    break
            if new_unknown is None:
                new_unknown = unknown_labels[-1]
            if new_unknown not in unknown_labels:
                unknown_labels.append(new_unknown)

            # Decide which model to use for estimation: if updating is enabled and an unknown model exists, prefer it;
            # otherwise use fallback known class model
            Theta_use = None
            if update_model and manager.exists(new_unknown):
                Theta_use = manager.load(new_unknown)
            else:
                Theta_use = manager.load(fallback)
                if Theta_use is None:
                    # Train from merged group corresponding to fallback
                    group_for_fallback: List[str] = []
                    for grp in merged_trainlist:
                        if extract_condition_prefix(grp) == fallback:
                            group_for_fallback = grp
                            break
                    if not group_for_fallback:
                        group_for_fallback = coarse_to_names.get(fallback, [])
                    if not group_for_fallback:
                        print(f'Warning: no training data found for fallback class {fallback}. Skipping estimation.')
                        continue
                    Theta_use = manager.train_from_files(fallback, group_for_fallback, piece='all')

            Y_pred = manager.estimate(Theta_use, U_test)

            # Train or update unknown model (only when update_model is True)
            if update_model:
                if not manager.exists(new_unknown):
                    model_tmp = MIMOFIR(past_order, future_order, channel_in, channel_out, mean_flag, test_names)
                    Theta_unknown = model_tmp.fit(piece='all')
                    manager.save(new_unknown, Theta_unknown)
                else:
                    Theta_curr = manager.load(new_unknown)
                    Theta_updated = manager.update_with_new_data(Theta_curr, U_test, Y_test, alpha=0.2)
                    manager.save(new_unknown, Theta_updated)

            # Update EVM with this new unknown sample's features if supported and updating enabled
            if update_model:
                try:
                    feat_unknown = compute_feature_for_name(test_name)
                    if hasattr(recognizer, 'add_class_samples'):
                        recognizer.add_class_samples(new_unknown, feat_unknown.reshape(1, -1))
                except Exception as e_update_evm:
                    print(f'Warning: failed to update recognizer with unknown sample: {e_update_evm}')

            print(f'Assigned label: {new_unknown}. Unknown pool: {unknown_labels}')

            # Save prediction
            safe_test_dir = _sanitize_filename(test_name)
            mode_dir = 'update' if update_model else 'unupdate'
            out_dir = os.path.join(results_dir, safe_test_dir, mode_dir)
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
                # train from existing known data group (merged)
                for group in merged_trainlist:
                    if extract_condition_prefix(group) == cls_hat:
                        Theta = manager.train_from_files(cls_hat, group, piece='all')
                        break
            Y_pred = manager.estimate(Theta, U_test)

            # update model with new data (only when update_model is True)
            if update_model:
                Theta_updated = manager.update_with_new_data(Theta, U_test, Y_test, alpha=0.2)
                manager.save(cls_hat, Theta_updated)

            # Save prediction
            safe_test_dir = _sanitize_filename(test_name)
            mode_dir = 'update' if update_model else 'unupdate'
            out_dir = os.path.join(results_dir, safe_test_dir, mode_dir)
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


def _sanitize_filename(name: str) -> str:
    # Keep letters, numbers, underscore, hyphen, and dot
    return re.sub(r'[^A-Za-z0-9._-]', '_', name).strip(' .')


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
    safe_base = _sanitize_filename(test_name + '_ieee')
    png_path = os.path.abspath(os.path.join(out_dir, f'{safe_base}.png'))
    pdf_path = os.path.abspath(os.path.join(out_dir, f'{safe_base}.pdf'))
    # Robust PNG save
    try:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    except Exception as e_png:
        try:
            with open(png_path, 'wb') as fh:
                fig.savefig(fh, format='png', dpi=300, bbox_inches='tight')
        except Exception as e_png2:
            alt_png = os.path.abspath(os.path.join(out_dir, 'plot.png'))
            try:
                fig.savefig(alt_png, dpi=300, bbox_inches='tight')
            except Exception as e_png3:
                print(f'Warning: failed to save PNG to {png_path}: {e_png}; fallback {alt_png} failed: {e_png2}; final error: {e_png3}')
    # Robust PDF save
    try:
        fig.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
    except Exception as e_pdf:
        try:
            with open(pdf_path, 'wb') as fh:
                fig.savefig(fh, format='pdf', dpi=600, bbox_inches='tight')
        except Exception as e_pdf2:
            alt_pdf = os.path.abspath(os.path.join(out_dir, 'plot.pdf'))
            try:
                fig.savefig(alt_pdf, format='pdf', dpi=600, bbox_inches='tight')
            except Exception as e_pdf3:
                print(f'Warning: failed to save PDF to {pdf_path}: {e_pdf}; fallback {alt_pdf} failed: {e_pdf2}; final error: {e_pdf3}')
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
    safe_base = _sanitize_filename('combined_ieee')
    png_path = os.path.abspath(os.path.join(results_dir, f'{safe_base}.png'))
    pdf_path = os.path.abspath(os.path.join(results_dir, f'{safe_base}.pdf'))
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    channel_in = [69, 78, 83, 84, 91, 94, 95, 96]
    channel_out = [32, 33, 34, 35, 36, 37]
    past_order = 50
    future_order = 50
    mean_flag = True

    EXP = 3
    UPDATE_MODEL = False

    if EXP == 1:
        trainlist = [
            ['BC30_01', 'BC30_02', 'BC30_03'],
            ['RE30_01', 'RE30_02', 'RE30_03'],
            ['CJ30_01', 'CJ30_02', 'CJ30_03'],
        ]
        testlist = [
            ['BC50_01'], ['BC50_02'], ['RC30_01'], ['RC30_02']
        ]
    elif EXP == 2:
        trainlist = [
            ['BC30_01', 'BC30_02', 'BC30_03', 'BC40_01', 'BC40_02', 'BC40_03', 'BC50_01', 'BC50_02', 'BC50_03'],
            ['RE20_01', 'RE20_02', 'RE20_03', 'RE30_01', 'RE30_02', 'RE30_03', 'RE40_01', 'RE40_02', 'RE40_03'],
            ['CJ30_01', 'CJ30_02', 'CJ30_03', 'CJ40_01', 'CJ40_02', 'CJ40_03', 'CJ50_01', 'CJ50_02', 'CJ50_03'],
            ['ST25_01', 'ST25_02', 'ST25_03', 'ST40_01', 'ST40_02', 'ST40_03', 'ST50_01', 'ST50_02', 'ST50_03'],
            ['DP10_01', 'DP10_02', 'DP10_03', 'DP20_01', 'DP20_02', 'DP20_03', 'DP30_01', 'DP30_02', 'DP30_03'],
            ['CS10_01', 'CS10_02', 'CS10_03', 'CS20_01', 'CS20_02', 'CS20_03', 'CS30_01', 'CS30_02', 'CS30_03'],
        ]
        testlist = [
            ['BC30_01'], ['BC30_02'], ['BC30_03'], ['BC40_01'], ['BC40_02'], ['BC40_03'], ['BC50_01'], ['BC50_02'], ['BC50_03'],
            ['RE20_01'], ['RE20_02'], ['RE20_03'], ['RE30_01'], ['RE30_02'], ['RE30_03'], ['RE40_01'], ['RE40_02'], ['RE40_03'],
            ['CJ30_01'], ['CJ30_02'], ['CJ30_03'], ['CJ40_01'], ['CJ40_02'], ['CJ40_03'], ['CJ50_01'], ['CJ50_02'], ['CJ50_03'],
            ['ST25_01'], ['ST25_02'], ['ST25_03'], ['ST40_01'], ['ST40_02'], ['ST40_03'], ['ST50_01'], ['ST50_02'], ['ST50_03'],
            ['DP10_01'], ['DP10_02'], ['DP10_03'], ['DP20_01'], ['DP20_02'], ['DP20_03'], ['DP30_01'], ['DP30_02'], ['DP30_03'],
            ['CS10_01'], ['CS10_02'], ['CS10_03'], ['CS20_01'], ['CS20_02'], ['CS20_03'], ['CS30_01'], ['CS30_02'], ['CS30_03'],
            ['RC30_01'], ['RC30_02'], ['RC30_03'], ['RC40_01'], ['RC40_02'], ['RC40_03'], ['RC50_01'], ['RC50_02'], ['RC50_03'],
        ]
    elif EXP == 3:
        trainlist = [
            ['BC30_01', 'BC30_02', 'BC40_01', 'BC40_02', 'BC50_01', 'BC50_02'], ['BK30_01', 'BK50_01', 'BK50_02', 'BK70_01', 'BK70_02'], ['BR20_01', 'BR20_02', 'BR30_01', 'BR30_02', 'BR40_01', 'BR40_02'],
            ['BTW_01', 'BTW_02', 'BTW_03', 'BTW_04'], ['CBFP40_01', 'CBFP40_02', 'CBFP50_01', 'CBFP50_02'], ['CJ30_01', 'CJ30_02', 'CJ40_01', 'CJ40_02', 'CJ50_01', 'CJ50_02'],
            ['CR05_01', 'CR05_02', 'CR15_01', 'CR15_02', 'CR15_03', 'CR15_04', 'CR25_01', 'CR25_02', 'CR25_03'], ['CS10_01', 'CS10_02', 'CS20_01', 'CS20_02', 'CS30_01', 'CS30_02'], 
            ['DP10_01', 'DP10_02', 'DP20_01', 'DP20_02', 'DP30_01', 'DP30_02'], ['PH20_01', 'PH20_02', 'PH30_01', 'PH30_02', 'PH40_01', 'PH40_02'],
            ['RC30_01', 'RC30_02', 'RC40_01', 'RC40_02', 'RC40_03', 'RC40_04', 'RC50_01', 'RC50_02', 'RC50_03', 'RC50_04'], ['RE20_01', 'RE20_02', 'RE30_01', 'RE30_02', 'RE40_01', 'RE40_02'],
            ['RP30_01', 'RP30_02', 'RP40_01', 'RP40_02', 'RP50_01', 'RP50_02'], ['SG30_01', 'SG30_02', 'SG30_03', 'SG40_01', 'SG40_02', 'SG50_01', 'SG50_02'], ['SGHB_01', 'SGHB_02'],
            ['SGLB_01', 'SGLB_02'], ['SGMB_01', 'SGMB_02'], ['ST25_01', 'ST25_02', 'ST40_01', 'ST40_02', 'ST50_01', 'ST50_02'], ['SW20_01', 'SW20_02', 'SW25_01', 'SW25_02', 'SW35_01', 'SW35_02'], 
            ['TW05_01', 'TW05_02', 'TW15_01', 'TW15_02', 'TW25_01', 'TW25_02'], ['WB30_01', 'WB30_02', 'WB40_01', 'WB40_02', 'WB50_01', 'WB50_02'],
            ['WBA40_01', 'WBA40_02', 'WBA40_03', 'WBA40_04', 'WBA60_01', 'WBA60_02', 'WBA60_03', 'WBA80_01', 'WBA80_02'], ['WBHB_01', 'WBHB_02'], ['WBLB_01', 'WBLB_02'], ['WBMB_01', 'WBMB_02']
        ]
        testlist = [
            ['SW20_03', 'SW25_03', 'SW35_03'，'RE40_01', 'RE40_02' ]
        ]
    else:
        trainlist = [
            ['SW20_01', 'SW20_02', 'SW25_01', 'SW25_02', 'SW35_01', 'SW35_02'],
            ['RE20_01', 'RE20_02', 'RE30_01', 'RE30_02', 'RE40_01', 'RE40_02'],
            ['CJ30_01', 'CJ30_02', 'CJ40_01', 'CJ40_02', 'CJ50_01', 'CJ50_02'],
        ]
        testlist = [
            ['SW20_03'], ['SW25_03'], ['SW35_03'],
            ['RE20_03'], ['RE30_03'], ['RE40_03'],
            ['CJ30_03'], ['CJ40_03'], ['CJ50_03'],
        ]


    # trainlist = [
    #         ['BC30_01', 'BC30_02', 'BC40_01', 'BC40_02', 'BC50_01', 'BC50_02'], ['BK30_01', 'BK50_01', 'BK50_02', 'BK70_01', 'BK70_02'], ['BR20_01', 'BR20_02', 'BR30_01', 'BR30_02', 'BR40_01', 'BR40_02'],
    #         ['BTW_01', 'BTW_02', 'BTW_03', 'BTW_04'], ['CBFP40_01', 'CBFP40_02', 'CBFP50_01', 'CBFP50_02'], ['CJ30_01', 'CJ30_02', 'CJ40_01', 'CJ40_02', 'CJ50_01', 'CJ50_02'],
    #         ['CR05_01', 'CR05_02', 'CR15_01', 'CR15_02', 'CR15_03', 'CR15_04', 'CR25_01', 'CR25_02', 'CR25_03'], ['CS10_01', 'CS10_02', 'CS20_01', 'CS20_02', 'CS30_01', 'CS30_02'], 
    #         ['DP10_01', 'DP10_02', 'DP20_01', 'DP20_02', 'DP30_01', 'DP30_02'], ['PH20_01', 'PH20_02', 'PH30_01', 'PH30_02', 'PH40_01', 'PH40_02'],
    #         ['RC30_01', 'RC30_02', 'RC40_01', 'RC40_02', 'RC40_03', 'RC40_04', 'RC50_01', 'RC50_02', 'RC50_03', 'RC50_04'], ['RE20_01', 'RE20_02', 'RE30_01', 'RE30_02', 'RE40_01', 'RE40_02'],
    #         ['RP30_01', 'RP30_02', 'RP40_01', 'RP40_02', 'RP50_01', 'RP50_02'], ['SG30_01', 'SG30_02', 'SG30_03', 'SG40_01', 'SG40_02', 'SG50_01', 'SG50_02'], ['SGHB_01', 'SGHB_02'],
    #         ['SGLB_01', 'SGLB_02'], ['SGMB_01', 'SGMB_02'], ['ST25_01', 'ST25_02', 'ST40_01', 'ST40_02', 'ST50_01', 'ST50_02'], ['SW20_01', 'SW20_02', 'SW25_01', 'SW25_02', 'SW35_01', 'SW35_02'], 
    #         ['TW05_01', 'TW05_02', 'TW15_01', 'TW15_02', 'TW25_01', 'TW25_02'], ['WB30_01', 'WB30_02', 'WB40_01', 'WB40_02', 'WB50_01', 'WB50_02'],
    #         ['WBA40_01', 'WBA40_02', 'WBA40_03', 'WBA40_04', 'WBA60_01', 'WBA60_02', 'WBA60_03', 'WBA80_01', 'WBA80_02'], ['WBHB_01', 'WBHB_02'], ['WBLB_01', 'WBLB_02'], ['WBMB_01', 'WBMB_02']
    #     ]
    # testlist = [
    #         ['BC30_03'], ['BC40_03'], ['BC50_03'], ['BK30_02'], ['BK50_03'], ['BK70_03'], ['BR20_03'], ['BR30_03'], ['BR40_03'], ['BTW_05'], ['CBFP40_03'], ['CBFP50_03'], ['CJ30_03'], ['CJ40_03'],
    #         ['CJ50_03'], ['CR05_03'], ['CR15_05'], ['CR25_04'], ['CS10_03'], ['CS20_03'], ['CS30_03'], ['DP10_03'], ['DP20_03'], ['DP30_03'], ['PH20_03'], ['PH30_03'], ['PH40_03'], ['RC30_03'],
    #         ['RC40_05'], ['RC50_05'], ['RE20_03'], ['RE30_03'], ['RE40_03'], ['RP30_03'], ['RP40_03'], ['RP50_03'], ['SG30_04'], ['SG40_03'], ['SG50_03'], ['SGHB_03'], ['SGLB_03'], ['SGMB_03'],
    #         ['ST25_03'], ['ST40_03'], ['ST50_03'], ['SW20_03'], ['SW25_03'], ['SW35_03'], ['TW05_03'], ['TW15_03'], ['TW25_03'], ['WB30_03'], ['WB40_03'], ['WB50_03'], ['WBA40_05'], ['WBA60_04'],
    #         ['WBA80_03'], ['WBHB_03'], ['WBLB_03'], ['WBMB_03']
    #     ]

    # Select results directory based on EXP
    if EXP == 1:
        results_dir = os.path.join('result', 'EXP1')
    else:
        results_dir = os.path.join('result', 'pipeline')

    pipeline(trainlist, testlist, channel_in, channel_out, past_order, future_order, mean_flag, update_model=UPDATE_MODEL, results_dir=results_dir)


