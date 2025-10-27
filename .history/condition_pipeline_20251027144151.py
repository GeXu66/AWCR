import os
import numpy as np
import scipy.io as scio
from typing import List, Dict

from feature_library import build_feature_library, compute_feature_for_name, extract_condition_prefix
from evm_classifier import EVMClassifier
from models_manager import MIMOFIRManager
from MIMOFIR import MIMOFIR


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

    # 1) Build feature library for known conditions
    saved = build_feature_library(trainlist, feature_dir=feature_dir)
    class_to_features: Dict[str, np.ndarray] = {}
    for group in trainlist:
        prefix = extract_condition_prefix(group)
        F = np.load(os.path.join(feature_dir, f'{prefix}.npy'))
        class_to_features[prefix] = F

    # 2) Train EVM
    evm = EVMClassifier(tail_frac=0.5, reject_threshold=0.5)
    evm.fit(class_to_features)

    # 3) Prepare MIMOFIR manager
    manager = MIMOFIRManager(past_order, future_order, channel_in, channel_out, mean_flag, models_dir=models_dir)

    # 4) Ensure models exist for known classes
    for group in trainlist:
        prefix = extract_condition_prefix(group)
        if not manager.exists(prefix):
            manager.train_from_files(prefix, group, piece='all')

    # unknown pool management
    unknown_labels: List[str] = []

    # 5) Sequentially process test conditions
    for test_names in testlist:
        test_name = test_names[0]
        feat = compute_feature_for_name(test_name)
        cls_hat, prob, scores = evm.predict(feat)

        # Read input/output for this test
        U_test, Y_test = read_io_for_names(test_names, channel_in, channel_out, mean_flag)

        if cls_hat == 'Unknown':
            # choose most similar known class as fallback for estimation
            if len(scores) == 0:
                # if completely empty, skip
                continue
            fallback = max(scores, key=scores.get)
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

            # Save prediction
            out_dir = os.path.join(results_dir, test_name)
            os.makedirs(out_dir, exist_ok=True)
            scio.savemat(os.path.join(out_dir, 'prediction.mat'), {'Y_true': Y_test, 'Y_pred': Y_pred, 'class': new_unknown})
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
        ['SW20_03'], ['SW25_03'], ['SW35_03'], ['RE20_03'], ['RE30_03'], ['RE40_03'],
        ['CJ30_03'], ['CJ40_03'], ['CJ50_03'],
    ]

    pipeline(trainlist, testlist, channel_in, channel_out, past_order, future_order, mean_flag)


