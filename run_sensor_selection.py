import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the RobustSensorSelection class
from robust_sensor_selection import RobustSensorSelection


def preprocess_data(data_folder, file_list):
    """
    Preprocess data from .mat files for easier use.

    Parameters:
    -----------
    data_folder : str
        Folder containing .mat files
    file_list : list
        List of filenames to process

    Returns:
    --------
    processed_data : dict
        Dictionary containing processed data
    """
    processed_data = {}

    for filename in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(data_folder, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue

        # Load data
        try:
            mat_data = sio.loadmat(file_path)

            # Extract data - adjust variable name if needed
            if 'data' in mat_data:
                data = mat_data['data']
            else:
                # If 'data' not found, try to find the main data array
                for key in mat_data:
                    if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2:
                        if mat_data[key].shape[0] > 100:  # Assume it's time series data
                            data = mat_data[key]
                            break

            # Store in dictionary, use filename without extension as key
            base_name = os.path.splitext(filename)[0]
            processed_data[base_name] = data

            print(f"Processed {filename}: Shape {data.shape}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return processed_data


def load_data_with_validation(rss, data_folder, conditions):
    """
    Load data for conditions with proper validation.

    Parameters:
    -----------
    rss : RobustSensorSelection
        RobustSensorSelection object
    data_folder : str
        Folder containing data files
    conditions : list
        List of base condition names (e.g., ['BC30', 'BC40', 'BC50'])

    Returns:
    --------
    loaded_conditions : list
        List of condition names actually loaded
    """
    loaded_conditions = []

    for base_condition in conditions:
        # Identify all files for this condition
        condition_files = []
        for file in os.listdir(data_folder):
            if file.startswith(base_condition) and file.endswith('.mat'):
                condition_files.append(file)

        condition_files.sort()

        if len(condition_files) == 0:
            print(f"No files found for condition {base_condition}, skipping.")
            continue

        # Use the first files for training and the last one for validation
        training_files = condition_files[:-1]
        validation_file = condition_files[-1]

        if len(training_files) == 0:
            print(f"Not enough files for condition {base_condition}, using the only file for training.")
            training_files = [validation_file]
            validation_file = None

        # Load training data
        for i, file in enumerate(training_files):
            file_path = os.path.join(data_folder, file)
            condition_name = f"{base_condition}_{i + 1:02d}"
            rss.load_data(file_path, condition_name)
            loaded_conditions.append(condition_name)
            print(f"Loaded training data for {condition_name} from {file}")

        # Load validation data if available
        if validation_file and validation_file != training_files[0]:
            file_path = os.path.join(data_folder, validation_file)
            for condition_name in loaded_conditions:
                if condition_name.startswith(base_condition):
                    rss.load_data(file_path, condition_name, validation=True)
                    print(f"Loaded validation data for {condition_name} from {validation_file}")

    return loaded_conditions

    # Solve the robust sensor selection problem
    print(f"Solving robust sensor selection for {p} sensors across {len(conditions)} conditions")
    z_opt, gamma_opt = rss.solve_robust_selection(conditions, p, relaxed=True)
    print("Optimization complete!")

    # Get selected channels
    selected_indices = np.where(z_opt > 0.5)[0]
    selected_channels = [channel_in[i] for i in selected_indices]

    print(f"Selected {len(selected_indices)} out of {len(channel_in)} sensors:")
    for i, idx in enumerate(selected_indices):
        print(f"  {i + 1}. Sensor {idx}: Channel {channel_in[idx]}")

    # Plot results
    rss.plot_selection_results(z_opt, conditions)

    # Compare with baseline
    rss.compare_with_baseline(z_opt, conditions, num_baselines=10)

    return z_opt, selected_channels


def evaluate_on_new_conditions(rss, z_opt, new_conditions, data_folder):
    """
    Evaluate the selected sensors on new operating conditions.

    Parameters:
    -----------
    rss : RobustSensorSelection
        Initialized RobustSensorSelection object
    z_opt : numpy.ndarray
        Optimal sensor selection vector
    new_conditions : list
        List of new condition names to evaluate on
    data_folder : str
        Folder containing data files

    Returns:
    --------
    performance : dict
        Dictionary of MSE values for each condition
    """
    performance = {}

    # Load and evaluate on new conditions
    for condition in new_conditions:
        file_path = os.path.join(data_folder, f'{condition}.mat')

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue

        # Load new condition
        rss.load_data(file_path, condition)

        # Evaluate performance
        mse = rss.evaluate_selection(z_opt, condition)
        performance[condition] = mse

        print(f"Performance on {condition}: MSE = {mse:.6f}")

    return performance


def run_sensor_selection(data_folder, conditions, channel_in, channel_out, p=20, r=3, d=3):
    """
    Run the robust sensor selection algorithm on the given data.

    Parameters:
    -----------
    data_folder : str
        Folder containing .mat files
    conditions : list
        List of base condition names (e.g., ['BC30', 'BC40', 'BC50'])
    channel_in : list
        List of input channel indices
    channel_out : list
        List of output channel indices
    p : int
        Number of sensors to select
    r : int
        Number of future time steps in FIR model
    d : int
        Number of past time steps in FIR model

    Returns:
    --------
    z_opt : numpy.ndarray
        Optimal sensor selection vector
    selected_channels : list
        List of selected channel indices
    """
    # Create selection object
    rss = RobustSensorSelection(r=r, d=d)
    rss.set_channels(channel_in, channel_out)

    # Load data with proper validation setup
    loaded_conditions = load_data_with_validation(rss, data_folder, conditions)

    if len(loaded_conditions) == 0:
        raise ValueError("No valid conditions could be loaded. Check data folder and condition names.")

    # Solve the robust sensor selection problem
    print(f"Solving robust sensor selection for {p} sensors across {len(loaded_conditions)} conditions")
    z_opt, gamma_opt = rss.solve_robust_selection(loaded_conditions, p, relaxed=True)
    print("Optimization complete!")

    # Get selected channels
    selected_indices = np.where(z_opt > 0.5)[0]
    selected_channels = [channel_in[i] for i in selected_indices]

    print(f"Selected {len(selected_indices)} out of {len(channel_in)} sensors:")
    for i, idx in enumerate(selected_indices):
        print(f"  {i + 1}. Sensor {idx}: Channel {channel_in[idx]}")

    # Plot results
    rss.plot_selection_results(z_opt, loaded_conditions)

    # Compare with baseline
    rss.compare_with_baseline(z_opt, loaded_conditions, num_baselines=10)

    return z_opt, selected_channels


def main():
    """
    Main execution function.
    """
    # Define input and output channels according to the problem specification
    channel_in = list(range(0, 32)) + list(range(56, 139))
    channel_in = [x for x in channel_in if x not in [57, 65, 76, 79]]
    channel_out = [32, 33, 34, 35, 36, 37]

    # Define data folder and conditions
    data_folder = './'  # Adjust to your data path

    # Use base condition names (without specific numbers)
    # The function will automatically find all matching files and use appropriate training/validation split
    conditions = ['BC30', 'BC40', 'BC50']

    # Run sensor selection
    z_opt, selected_channels = run_sensor_selection(
        data_folder=data_folder,
        conditions=conditions,
        channel_in=channel_in,
        channel_out=channel_out,
        p=20,  # Number of sensors to select
        r=3,  # Future time steps
        d=3  # Past time steps
    )

    # Optionally evaluate on additional conditions
    additional_conditions = []  # Add more condition bases if available, e.g., ['BC60', 'BC70']

    if additional_conditions:
        # Create selection object with same parameters
        rss = RobustSensorSelection(r=3, d=3)
        rss.set_channels(channel_in, channel_out)

        # Load the original conditions data
        loaded_conditions = load_data_with_validation(rss, data_folder, conditions)

        # Load and evaluate additional conditions
        additional_loaded = load_data_with_validation(rss, data_folder, additional_conditions)

        # Evaluate performance on additional conditions
        performance = {}
        for condition in additional_loaded:
            mse = rss.evaluate_selection(z_opt, condition)
            performance[condition] = mse
            print(f"Performance on {condition}: MSE = {mse:.6f}")

        if performance:
            # Plot additional results
            plt.figure(figsize=(10, 6))
            plt.bar(performance.keys(), performance.values())
            plt.ylabel('Mean Squared Error')
            plt.xlabel('Condition')
            plt.title('Performance on Additional Conditions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Save the selected sensors to a file
    np.savetxt('selected_sensors.txt', selected_channels, fmt='%d')
    print(f"Selected sensors saved to 'selected_sensors.txt'")

    # Also save as CSV with more information
    with open('selected_sensors_detailed.csv', 'w') as f:
        f.write("Index,ChannelNumber\n")
        for i, channel in enumerate(selected_channels):
            f.write(f"{i + 1},{channel}\n")
    print(f"Detailed selected sensors saved to 'selected_sensors_detailed.csv'")

    return z_opt, selected_channels


if __name__ == "__main__":
    main()
