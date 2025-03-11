import numpy as np
import torch
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from pathlib import Path
import multiprocessing
from functools import partial

# Set matplotlib font and style for IEEE Transaction papers
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13
matplotlib.rcParams['legend.fontsize'] = 12


def is_valid_tuple(input_data):
    if isinstance(input_data, tuple) and len(input_data) == 2:
        start, end = input_data
        if isinstance(start, int) and isinstance(end, int) and start > 0 and end > 0 and start < end:
            return True
    return False


def plot_comparison(traditional_predict, best_predict, real_out, channel_idx=0, condition_name='', output_names=None, save_path=None):
    """
    Plot both traditional and best channel predictions against real values on the same graph
    """
    # Define force/moment names if not provided
    if output_names is None:
        output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Get axis labels based on output channel
    output_name = output_names[channel_idx]
    y_label = get_output_label(output_name)

    # Extract the specific channel data
    if len(real_out.shape) > 1 and real_out.shape[1] > 1:
        real_out_channel = real_out[:, channel_idx]
        trad_predict_channel = traditional_predict[:, channel_idx]
        best_predict_channel = best_predict[:, channel_idx]
    else:
        real_out_channel = np.ravel(real_out)
        trad_predict_channel = np.ravel(traditional_predict)
        best_predict_channel = np.ravel(best_predict)

    # Create figure with IEEE Transaction style
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    cut_range = (0.1, 0.8)
    start = int(len(real_out_channel) * cut_range[0])
    end = int(len(real_out_channel) * cut_range[1])
    real_out_channel = real_out_channel[start:end]
    trad_predict_channel = trad_predict_channel[start:end]
    best_predict_channel = best_predict_channel[start:end]
    x = np.arange(1, len(real_out_channel) + 1)
    x = x / 500
    # 绘制曲线
    color1, color2, color3 = '#f6541f', '#2299f0', '#72b607'
    # Plot data with distinctive line styles
    ax.plot(x, real_out_channel, label='Measured', color=color1, linewidth=2, linestyle='-')
    ax.plot(x, trad_predict_channel, label='Empirical Sensors', color=color2, linewidth=2, linestyle='--')
    ax.plot(x, best_predict_channel, label='Optimized Sensors', color=color3, linewidth=2, linestyle=':')

    # Set formatting for IEEE Transaction style
    ax.spines['top'].set_linewidth(1.1)
    ax.spines['right'].set_linewidth(1.1)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)

    # Add labels
    ax.set_xlabel('Time (Seconds)', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Add legend in top-left corner
    ax.legend(loc='upper left', fontsize=12, framealpha=1, edgecolor='black')

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')

    plt.close(fig)


def plot_combined_comparison(traditional_predict, best_predict, real_out, condition_name, output_names=None, save_path=None):
    """
    Plot all six force/moment components in a single 6x1 figure

    Args:
        traditional_predict: Predictions from traditional sensor model
        best_predict: Predictions from optimized sensor model
        real_out: Real/measured values
        condition_name: Name of the test condition
        output_names: Names of the output channels
        save_path: Path where to save the combined figure
    """
    if output_names is None:
        output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Apply the same time range cut as in individual plots
    cut_range = (0.1, 0.8)
    start = int(real_out.shape[0] * cut_range[0])
    end = int(real_out.shape[0] * cut_range[1])

    real_out_cut = real_out[start:end, :]
    trad_predict_cut = traditional_predict[start:end, :]
    best_predict_cut = best_predict[start:end, :]

    # Generate time points
    x = np.arange(1, real_out_cut.shape[0] + 1)
    x = x / 500  # Convert to seconds

    # Define colors for consistent plotting
    color1, color2, color3 = '#f6541f', '#2299f0', '#72b607'

    # Create a large figure with 6 subplots (6 rows, 1 column)
    fig, axes = plt.subplots(6, 1, figsize=(8, 12), constrained_layout=True)

    # Plot each force/moment component
    for i, (ax, output_name) in enumerate(zip(axes, output_names)):
        # Extract data for this component
        real_data = real_out_cut[:, i]
        trad_data = trad_predict_cut[:, i]
        best_data = best_predict_cut[:, i]

        # Get appropriate y-axis label
        y_label = get_output_label(output_name)

        # Plot the data
        ax.plot(x, real_data, label='Measured', color=color1, linewidth=2, linestyle='-')
        ax.plot(x, trad_data, label='Empirical Sensors', color=color2, linewidth=2, linestyle='--')
        ax.plot(x, best_data, label='Optimized Sensors', color=color3, linewidth=2, linestyle=':')

        # Set formatting for IEEE Transaction style
        ax.spines['top'].set_linewidth(1.1)
        ax.spines['right'].set_linewidth(1.1)
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)

        # Add y-axis label
        ax.set_ylabel(y_label, fontsize=14)

        # Only add x-axis label to the bottom subplot
        if i == 5:  # Last subplot
            ax.set_xlabel('Time (Seconds)', fontsize=14)

        # Add legend to the first subplot only
        if i == 0:
            ax.legend(loc='upper left', fontsize=12, framealpha=1, edgecolor='black')

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')

    plt.close(fig)


def get_output_label(output_name):
    """Return appropriate axis label based on output channel name"""
    labels = {
        'Fx': 'Longitudinal Force (N)',
        'Fy': 'Lateral Force (N)',
        'Fz': 'Vertical Force (N)',
        'Mx': 'Rolling Moment (N·m)',
        'My': 'Pitching Moment (N·m)',
        'Mz': 'Yawing Moment (N·m)'
    }
    return labels.get(output_name, output_name)


def ridge_reg_torch(phi, y, alpha=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phi = torch.from_numpy(phi).to(device)
    y = torch.from_numpy(y).to(device)

    # Create augmented system for regularization
    n_samples, n_features = phi.shape

    # Direct ridge solution: (phi^T phi + alpha*I)^(-1) phi^T y
    regularized_solution = torch.linalg.solve(
        phi.T @ phi + alpha * torch.eye(n_features, device=device),
        phi.T @ y
    )

    return regularized_solution.cpu().numpy()


class FIR:
    def __init__(self, causal_order, non_causal_order, channel_in, channel_out, mean_flag, trainName):
        self.causal_order = causal_order
        self.non_causal_order = non_causal_order
        self.total_order = causal_order + non_causal_order + 1  # +1 for the current time step
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.phi = None
        self.theta = None
        self.ni = len(channel_in)
        self.no = len(channel_out)
        self.mean_flag = mean_flag
        self.input_length = self.total_order * self.ni

        # Handle multiple training files
        if isinstance(trainName, list):
            if all(isinstance(item, list) for item in trainName):
                # This is a list of lists
                self.trainfiles = []
                for sublist in trainName:
                    self.trainfiles.extend(['../Data/matdata/' + n + '.mat' for n in sublist])
            else:
                # This is a simple list
                self.trainfiles = ['../Data/matdata/' + n + '.mat' for n in trainName]
        else:
            self.trainfiles = ['../Data/matdata/' + trainName + '.mat']

    def read_mat(self, piece):
        train = []
        target = []
        for file in self.trainfiles:
            key = file.split('/')[-1]
            key = key.split('.')[-2]
            data = scio.loadmat(file)
            data = data[key]
            train.append(data[:, self.channel_in])
            target.append(data[:, self.channel_out])
            print('-------------------Loading Data-------------------')
            print(f'data {key} load successfully!')

        if len(train) > 1:
            train_all = train[0]
            target_all = target[0]
            for i in range(1, len(train)):
                train_all = np.vstack((train_all, train[i]))
                target_all = np.vstack((target_all, target[i]))
        else:
            train_all = train[0]
            target_all = target[0]

        if self.mean_flag:
            train_all = (train_all - np.mean(train_all, axis=0))
            target_all = (target_all - np.mean(target_all, axis=0))

        if is_valid_tuple(piece) and piece[1] <= train_all.shape[0]:
            self.time_length = piece[1] - piece[0] + 1
            self.train_all = train_all[piece[0]:piece[1] + 1, :]
            self.target_all = target_all[piece[0]:piece[1] + 1, :]
        elif piece == 'all':
            self.time_length = train_all.shape[0]
            self.train_all = train_all
            self.target_all = target_all
        else:
            print('input length wrong!')
            exit(0)

    def gen_phi(self):
        # Need to handle both past and future data for non-causal FIR
        # Trim some data points at the beginning and end for non-causal modeling
        effective_length = self.train_all.shape[0] - self.non_causal_order
        phi = np.zeros((effective_length, self.input_length))

        # For each effective time point
        for i in range(effective_length):
            # Current position in original data accounting for the non-causal offset
            current_pos = i + self.non_causal_order

            # Extract past, present, and future data for this time point
            time_window = np.zeros((self.total_order, self.ni))

            # Fill in the past data (causal part)
            for j in range(self.causal_order):
                pos = current_pos - (self.causal_order - j)
                if pos >= 0:
                    time_window[j] = self.train_all[pos]
                else:
                    # For positions before the start, repeat the first entry
                    time_window[j] = self.train_all[0]

            # Fill in the current time point
            time_window[self.causal_order] = self.train_all[current_pos]

            # Fill in the future data (non-causal part)
            for j in range(self.non_causal_order):
                pos = current_pos + j + 1
                if pos < self.train_all.shape[0]:
                    time_window[self.causal_order + j + 1] = self.train_all[pos]
                else:
                    # For positions after the end, repeat the last entry
                    time_window[self.causal_order + j + 1] = self.train_all[-1]

            # Flatten the time window into the feature vector
            phi[i, :] = time_window.T.reshape(1, -1)

        self.phi = phi
        # Adjust target data to match the effective length after non-causal windowing
        self.target_all = self.target_all[self.non_causal_order:self.non_causal_order + effective_length]

    def gen_phi_data(self, data, causal_order, non_causal_order):
        total_order = causal_order + non_causal_order + 1
        effective_length = data.shape[0] - non_causal_order
        phi_data = np.zeros((effective_length, data.shape[1] * total_order))

        # For each effective time point
        for i in range(effective_length):
            # Current position in original data accounting for the non-causal offset
            current_pos = i + non_causal_order

            # Extract past, present, and future data for this time point
            time_window = np.zeros((total_order, data.shape[1]))

            # Fill in the past data (causal part)
            for j in range(causal_order):
                pos = current_pos - (causal_order - j)
                if pos >= 0:
                    time_window[j] = data[pos]
                else:
                    # For positions before the start, repeat the first entry
                    time_window[j] = data[0]

            # Fill in the current time point
            time_window[causal_order] = data[current_pos]

            # Fill in the future data (non-causal part)
            for j in range(non_causal_order):
                pos = current_pos + j + 1
                if pos < data.shape[0]:
                    time_window[causal_order + j + 1] = data[pos]
                else:
                    # For positions after the end, repeat the last entry
                    time_window[causal_order + j + 1] = data[-1]

            # Flatten the time window into the feature vector
            phi_data[i, :] = time_window.T.reshape(1, -1)

        return phi_data

    def train(self):
        print('-------------------Training-------------------')
        self.theta = ridge_reg_torch(self.phi, self.target_all)
        return self.theta

    def predict(self, u):
        predict_scale = u.shape
        print('test input shape', predict_scale)
        phi_u = self.gen_phi_data(data=u, causal_order=self.causal_order, non_causal_order=self.non_causal_order)
        print('test input recurrent shape', phi_u.shape)
        print('-------------------Predicting-------------------')
        predict_out = phi_u @ self.theta

        # Return prediction and adjusted test data
        return predict_out, u[self.non_causal_order:self.non_causal_order + phi_u.shape[0]]


def compute_metric(real_out, predict_out):
    # Calculate metrics for each output channel
    num_outputs = real_out.shape[1]
    rmse_channels = np.zeros(num_outputs)
    mae_channels = np.zeros(num_outputs)
    r2_channels = np.zeros(num_outputs)
    mape_channels = np.zeros(num_outputs)

    for i in range(num_outputs):
        rmse_channels[i] = np.sqrt(mean_squared_error(real_out[:, i], predict_out[:, i]))
        mae_channels[i] = mean_absolute_error(real_out[:, i], predict_out[:, i])
        r2_channels[i] = r2_score(real_out[:, i], predict_out[:, i])
        mape_channels[i] = mean_absolute_percentage_error(real_out[:, i], predict_out[:, i])
        # # Calculate MAPE (Mean Absolute Percentage Error)
        # # Handle potential division by zero
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     abs_percentage_errors = np.abs((real_out[:, i] - predict_out[:, i]) / np.abs(real_out[:, i] + 1e-10)) * 100
        #     mape_channels[i] = np.mean(abs_percentage_errors[~np.isinf(abs_percentage_errors)])

    # Average metrics across all channels
    rmse_avg = np.mean(rmse_channels)
    mae_avg = np.mean(mae_channels)
    r2_avg = np.mean(r2_channels)
    mape_avg = np.mean(mape_channels)

    # Return both individual channel metrics and averages
    return {
        'rmse': rmse_channels,
        'mae': mae_channels,
        'r2': r2_channels,
        'mape': mape_channels,
        'rmse_avg': rmse_avg,
        'mae_avg': mae_avg,
        'r2_avg': r2_avg,
        'mape_avg': mape_avg
    }


def fir_prediction(causal_order, non_causal_order, channel_in, channel_out, mean_flag, trainName, testName,
                   data_volume, output_names=None):
    if output_names is None:
        output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Create FIR models for training and testing
    fir = FIR(causal_order, non_causal_order, channel_in, channel_out, mean_flag, trainName)
    test_fir = FIR(causal_order, non_causal_order, channel_in, channel_out, mean_flag, testName)

    # Process data
    fir.read_mat(data_volume)
    test_fir.read_mat(data_volume)
    fir.gen_phi()
    fir.train()

    # Get predictions and adjusted test data
    predict_out, adjusted_test_data = fir.predict(test_fir.train_all)

    # Adjust target data to match the prediction length
    real_out = test_fir.target_all[test_fir.non_causal_order:test_fir.non_causal_order + predict_out.shape[0]]

    # Compute metrics
    metrics = compute_metric(real_out, predict_out)

    # Prepare data for JSON
    result_data = {
        'input_channels': channel_in,
        'real_values': real_out.tolist(),
        'predicted_values': predict_out.tolist(),
        'metrics': {
            'rmse': metrics['rmse'].tolist(),
            'mape': metrics['mape'].tolist(),
            'r2': metrics['r2'].tolist(),
            'rmse_avg': float(metrics['rmse_avg']),
            'mape_avg': float(metrics['mape_avg']),
            'r2_avg': float(metrics['r2_avg'])
        }
    }

    return real_out, predict_out, metrics, result_data


def process_single_channel(channel, causal_order, non_causal_order, channel_out, mean_flag, train_list, test_list, data_volume):
    """Process a single input channel - used for parallel processing"""
    channel = [channel]
    print(f'Processing channel: {channel}')

    # Number of different operating conditions
    num_conditions = len(train_list)

    # Initialize metrics for this channel across all conditions
    channel_condition_metrics = np.zeros((num_conditions, 4))

    # Test this channel under all operating conditions
    for condition_idx in range(num_conditions):
        trainName = train_list[condition_idx]
        testName = test_list[condition_idx]

        # Create FIR model for this channel and condition
        fir = FIR(causal_order, non_causal_order, channel, channel_out, mean_flag, trainName)
        test_fir = FIR(causal_order, non_causal_order, channel, channel_out, mean_flag, testName)

        # Process data
        fir.read_mat(data_volume)
        test_fir.read_mat(data_volume)
        fir.gen_phi()
        fir.train()

        # Get predictions and adjusted test data
        predict_out, adjusted_test_data = fir.predict(test_fir.train_all)

        # Adjust target data to match the prediction length
        real_out = test_fir.target_all[test_fir.non_causal_order:test_fir.non_causal_order + predict_out.shape[0]]

        # Compute metrics for this condition
        metrics = compute_metric(real_out, predict_out)
        condition_metrics = [metrics['rmse_avg'], metrics['mae_avg'], metrics['r2_avg'], metrics['mape_avg']]
        channel_condition_metrics[condition_idx] = condition_metrics

    # Average metrics across all conditions for this channel
    avg_metrics = np.mean(channel_condition_metrics, axis=0)

    # Compute fusion metric (weighted combination of individual metrics)
    fusion_metric = 0.4 * avg_metrics[0] + 0.1 * avg_metrics[1] + 0.3 * avg_metrics[3] + 10 * (1 - avg_metrics[2])

    return {
        'channel': channel[0],
        'metrics': avg_metrics,
        'fusion_metric': fusion_metric
    }


def recurrent_analysis_multi_condition_parallel(causal_order, non_causal_order, channel_in, channel_out, mean_flag,
                                                train_list, test_list, data_volume, output_names=None, n_processes=None):
    """Parallel version of recurrent analysis for multiple conditions"""
    if output_names is None:
        output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Set the number of processes to use (default to CPU count)
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    print(f"Starting parallel processing with {n_processes} processes")

    # Create a partial function with fixed parameters
    process_func = partial(
        process_single_channel,
        causal_order=causal_order,
        non_causal_order=non_causal_order,
        channel_out=channel_out,
        mean_flag=mean_flag,
        train_list=train_list,
        test_list=test_list,
        data_volume=data_volume
    )

    # Process all channels in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(process_func, channel_in)

    # Extract results and sort by fusion metric
    channel_metrics = []
    for result in results:
        channel_metrics.append({
            'channel': result['channel'],
            'metrics': np.around(result['metrics'], decimals=3),
            'fusion_metric': np.around(result['fusion_metric'], decimals=3)
        })

    # Sort by fusion metric (ascending)
    channel_metrics.sort(key=lambda x: x['fusion_metric'])

    # Get the 20 best channels
    best_20_metrics = channel_metrics[:20]
    best_20_channels = [item['channel'] for item in best_20_metrics]

    print("Best 20 channels and metrics:")
    for i, item in enumerate(best_20_metrics):
        print(f"{i + 1}. Channel {item['channel']}: {item['fusion_metric']} (RMSE: {item['metrics'][0]}, R²: {item['metrics'][2]}, MAPE: {item['metrics'][3]}%)")

    return best_20_channels


def evaluate_and_plot_comparison(causal_order, non_causal_order, traditional_channels, best_channels, channel_out,
                                 mean_flag, train_list, test_list, data_volume, output_names=None):
    """Evaluate both traditional and best channels and create comparison plots and DPF files"""
    if output_names is None:
        output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Store results for JSON export
    all_results = {}

    # Process each condition
    for condition_idx in range(len(train_list)):
        trainName = train_list[condition_idx]
        testName = test_list[condition_idx]

        # Extract condition name (e.g., 'BC30' from 'BC30_03')
        condition_name = testName[0].split('_')[0]

        print(f"\nEvaluating condition {condition_name}:")

        # Get traditional channel predictions
        print("  Processing traditional channels...")
        trad_real_out, trad_predict_out, trad_metrics, trad_results = fir_prediction(
            causal_order, non_causal_order, traditional_channels, channel_out, mean_flag,
            trainName, testName, data_volume, output_names
        )

        # Get best channel predictions
        print("  Processing best channels...")
        best_real_out, best_predict_out, best_metrics, best_results = fir_prediction(
            causal_order, non_causal_order, best_channels, channel_out, mean_flag,
            trainName, testName, data_volume, output_names
        )

        # Print comparison
        print(f"  Traditional channels - RMSE: {trad_metrics['rmse_avg']:.4f}, R²: {trad_metrics['r2_avg']:.4f}, MAPE: {trad_metrics['mape_avg']:.4f}%")
        print(f"  Selected channels    - RMSE: {best_metrics['rmse_avg']:.4f}, R²: {best_metrics['r2_avg']:.4f}, MAPE: {best_metrics['mape_avg']:.4f}%")

        # Create directory for figures and DPF files
        fig_dir = Path(f'./result/sensor_selection/fig/{condition_name}')
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Plot individual comparisons for each output channel
        for i, output_name in enumerate(output_names):
            # Generate PDF plot
            fig_path = fig_dir / f'{condition_name}-{output_name}.pdf'
            plot_comparison(
                trad_predict_out, best_predict_out, trad_real_out,
                channel_idx=i, condition_name=condition_name,
                output_names=output_names, save_path=fig_path
            )

        # Create combined 6x1 plot with all force/moment components
        combined_fig_path = fig_dir / f'{condition_name}-combined.pdf'
        plot_combined_comparison(
            trad_predict_out, best_predict_out, trad_real_out,
            condition_name=condition_name, output_names=output_names,
            save_path=combined_fig_path
        )
        print(f"  Combined 6x1 figure created: {combined_fig_path}")

        # Add results to the JSON data
        for i, output_name in enumerate(output_names):
            all_results[f'{condition_name}-{output_name}'] = {
                'traditional_channels': traditional_channels,
                'selected_channels': best_channels,
                'real_values': trad_real_out[:, i].tolist(),  # Same for both methods
                'traditional_predicted': trad_predict_out[:, i].tolist(),
                'selected_predicted': best_predict_out[:, i].tolist(),
                'traditional_metrics': {
                    'rmse': float(trad_metrics['rmse'][i]),
                    'mape': float(trad_metrics['mape'][i]),
                    'r2': float(trad_metrics['r2'][i])
                },
                'selected_metrics': {
                    'rmse': float(best_metrics['rmse'][i]),
                    'mape': float(best_metrics['mape'][i]),
                    'r2': float(best_metrics['r2'][i])
                }
            }

    return all_results


def save_results_to_json(all_results, train_list, test_list):
    """Save all test results to a JSON file"""
    # Create file name from first and last test condition
    first_test = test_list[0][0].split('_')[0]  # Get 'BC30' from 'BC30_03'
    last_test = test_list[-1][0].split('_')[0]  # Get 'BC50' from 'BC50_03'
    file_name = f"{first_test}-{last_test}.json"

    # Create directory if it doesn't exist
    data_dir = Path('./result/sensor_selection/data/')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save data to JSON file
    with open(data_dir / file_name, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {data_dir / file_name}")


if __name__ == '__main__':
    # Define traditional channel inputs
    tradition_channel_in = [list(range(90, 106)), list(range(109, 112)), list(range(118, 121))]
    tradition_channel_in = [x for sublist in tradition_channel_in for x in sublist]

    # Define all available channels
    channel_in = [list(range(0, 32)), list(range(56, 139))]
    channel_in = [x for sublist in channel_in for x in sublist]
    remove_list = [57, 65, 76, 79]
    channel_in = [x for x in channel_in if x not in remove_list]

    # Define output channels and their names
    channel_out = [32, 33, 34, 35, 36, 37]
    output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Define FIR model parameters
    causal_order = 25  # r: past time steps (causal part)
    non_causal_order = 25  # d: future time steps (non-causal part)
    mean_flag = True
    data_volume = 'all'

    # Define multiple operating conditions
    Trainlist = [['BC30_01', 'BC30_02'], ['BC40_01', 'BC40_02'], ['BC50_01', 'BC50_02'], ['BK30_01'], ['BK50_01', 'BK50_02'], ['BK70_01', 'BK70_02'], ['BR20_01', 'BR20_02'], ['BR30_01', 'BR30_02'],
                 ['BR40_01', 'BR40_02'], ['BTW_01', 'BTW_02', 'BTW_03', 'BTW_04'], ['CBFP40_01', 'CBFP40_02'], ['CBFP50_01', 'CBFP50_02'], ['CJ30_01', 'CJ30_02'], ['CJ40_01', 'CJ40_02'],
                 ['CJ50_01', 'CJ50_02'], ['CR05_01', 'CR05_02'], ['CR15_01', 'CR15_02', 'CR15_03', 'CR15_04'], ['CR25_01', 'CR25_02', 'CR25_03'], ['CS10_01', 'CS10_02'], ['CS20_01', 'CS20_02'],
                 ['CS30_01', 'CS30_02'], ['DP10_01', 'DP10_02'], ['DP20_01', 'DP20_02'], ['DP30_01', 'DP30_02'], ['PH20_01', 'PH20_02'], ['PH30_01', 'PH30_02'], ['PH40_01', 'PH40_02'],
                 ['RC30_01', 'RC30_02'], ['RC40_01', 'RC40_02', 'RC40_03', 'RC40_04'], ['RC50_01', 'RC50_02', 'RC50_03', 'RC50_04'], ['RE20_01', 'RE20_02'], ['RE30_01', 'RE30_02'],
                 ['RE40_01', 'RE40_02'],
                 ['RP30_01', 'RP30_02'], ['RP40_01', 'RP40_02'], ['RP50_01', 'RP50_02'], ['SG30_01', 'SG30_02', 'SG30_03'], ['SG40_01', 'SG40_02'], ['SG50_01', 'SG50_02'], ['SGHB_01', 'SGHB_02'],
                 ['SGLB_01', 'SGLB_02'], ['SGMB_01', 'SGMB_02'], ['ST25_01', 'ST25_02'], ['ST40_01', 'ST40_02'], ['ST50_01', 'ST50_02'], ['SW20_01', 'SW20_02'], ['SW25_01', 'SW25_02'],
                 ['SW35_01', 'SW35_02'], ['TW05_01', 'TW05_02'], ['TW15_01', 'TW15_02'], ['TW25_01', 'TW25_02'], ['WB30_01', 'WB30_02'], ['WB40_01', 'WB40_02'], ['WB50_01', 'WB50_02'],
                 ['WBA40_01', 'WBA40_02', 'WBA40_03', 'WBA40_04'], ['WBA60_01', 'WBA60_02', 'WBA60_03'], ['WBA80_01', 'WBA80_02'], ['WBHB_01', 'WBHB_02'], ['WBLB_01', 'WBLB_02'],
                 ['WBMB_01', 'WBMB_02']]
    Testlist = [['BC30_03'], ['BC40_03'], ['BC50_03'], ['BK30_02'], ['BK50_03'], ['BK70_03'], ['BR20_03'], ['BR30_03'], ['BR40_03'], ['BTW_05'], ['CBFP40_03'], ['CBFP50_03'], ['CJ30_03'], ['CJ40_03'],
                ['CJ50_03'], ['CR05_03'], ['CR15_05'], ['CR25_04'], ['CS10_03'], ['CS20_03'], ['CS30_03'], ['DP10_03'], ['DP20_03'], ['DP30_03'], ['PH20_03'], ['PH30_03'], ['PH40_03'], ['RC30_03'],
                ['RC40_05'], ['RC50_05'], ['RE20_03'], ['RE30_03'], ['RE40_03'], ['RP30_03'], ['RP40_03'], ['RP50_03'], ['SG30_04'], ['SG40_03'], ['SG50_03'], ['SGHB_03'], ['SGLB_03'], ['SGMB_03'],
                ['ST25_03'], ['ST40_03'], ['ST50_03'], ['SW20_03'], ['SW25_03'], ['SW35_03'], ['TW05_03'], ['TW15_03'], ['TW25_03'], ['WB30_03'], ['WB40_03'], ['WB50_03'], ['WBA40_05'], ['WBA60_04'],
                ['WBA80_03'], ['WBHB_03'], ['WBLB_03'], ['WBMB_03']]

    # Find best channels using parallel processing
    print("\nFinding best channels using parallel processing:")
    best_20_channel_num = recurrent_analysis_multi_condition_parallel(
        causal_order, non_causal_order, channel_in, channel_out, mean_flag,
        Trainlist, Testlist, data_volume, output_names
    )

    # Evaluate both traditional and best channels, create comparison plots
    print("\nEvaluating performance and creating comparison plots:")
    all_results = evaluate_and_plot_comparison(
        causal_order, non_causal_order, tradition_channel_in, best_20_channel_num,
        channel_out, mean_flag, Trainlist, Testlist, data_volume, output_names
    )

    # Save all results to JSON
    save_results_to_json(all_results, Trainlist, Testlist)
