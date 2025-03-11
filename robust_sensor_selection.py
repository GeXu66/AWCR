"""
Robust Sensor Selection for Wheel Force Estimation.
A complete implementation of the algorithm described in the paper.
"""
import json
import numpy as np
import scipy.io as sio
import scipy.linalg as la
from scipy import sparse
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class RobustSensorSelection:
    """
    Implementation of Cross-Coupled Multiplicative-Robust Sensor Selection for 
    wheel force estimation across multiple operating conditions.
    """

    def __init__(self, r=3, d=3):
        """
        Initialize the RobustSensorSelection class.

        Parameters:
        -----------
        r : int
            Number of future time steps in FIR model
        d : int
            Number of past time steps in FIR model
        """
        self.r = r  # Future time steps
        self.d = d  # Past time steps
        self.data = {}  # Store data for each condition
        self.uncertainty_bounds = {}  # Store uncertainty bounds
        self.noise_covariance = {}  # Store noise covariance
        self.channel_in = None  # Input channels
        self.channel_out = None  # Output channels

    def load_data(self, file_path, condition_name):
        """
        Load data from .mat file for a specific operating condition.

        Parameters:
        -----------
        file_path : str
            Path to the .mat file
        condition_name : str
            Name identifier for the operating condition
        validation : bool
            Whether this is validation data
        """
        print(f"Loading data for condition: {condition_name}")
        mat_data = sio.loadmat(file_path)

        # Extract data from the mat file
        # Assuming the mat file contains a variable 'data'
        if 'data' in mat_data:
            data = mat_data['data']
        else:
            # If data is not directly available, use the first variable that's a numpy array
            for key in mat_data:
                if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) >= 2:
                    data = mat_data[key]
                    break

        self.data[condition_name] = data

        return data

    def set_channels(self, channel_in, channel_out):
        """
        Set input and output channels.

        Parameters:
        -----------
        channel_in : list
            List of input channel indices
        channel_out : list
            List of output channel indices
        """
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.S = len(channel_in)  # Total number of sensors
        self.output_dim = len(channel_out)  # Number of output dimensions

        print(f"Set {self.S} input channels and {self.output_dim} output channels")

    def prepare_mimo_fir_data(self, condition):
        """
        Prepare MIMO-FIR data matrices for a specific condition.

        Parameters:
        -----------
        condition : str
            Name of the condition to prepare data for

        Returns:
        --------
        Phi : numpy.ndarray
            Measurement matrix
        Y_O : numpy.ndarray
            Output matrix
        """
        if condition not in self.data:
            raise ValueError(f"Condition {condition} not found in loaded data")

        data = self.data[condition]

        # Extract input and output data
        y_I = data[:, self.channel_in]  # Input measurements
        y_O = data[:, self.channel_out]  # Output measurements

        m = data.shape[0]  # Number of samples
        h = (self.r + self.d + 1) * self.S  # Total number of parameters

        # Prepare output matrix
        Y_O = y_O[self.r:(m - self.d), :]

        # Prepare measurement matrix (without sensor selection)
        Phi = np.zeros((m - self.d - self.r, h))

        for t in range(self.r, m - self.d):
            row_idx = t - self.r
            for s in range(self.S):
                col_idx_start = s * (self.r + self.d + 1)
                # For each sensor, collect measurements from t+d to t-r
                sensor_data = y_I[(t - self.r):(t + self.d + 1), s]
                Phi[row_idx, col_idx_start:(col_idx_start + self.r + self.d + 1)] = sensor_data[::-1]  # Reverse to match the paper's notation

        return Phi, Y_O

    def estimate_noise_covariance(self, condition, validation_condition=None):
        """
        Estimate noise covariance matrix for a condition using validation data.

        Parameters:
        -----------
        condition : str
            Name of the condition
        validation_condition : str, optional
            Name of the validation condition. If None, uses same condition name with '_validation' suffix
        """
        if validation_condition is None:
            validation_condition = condition + '_validation'

        if validation_condition not in self.data:
            raise ValueError(f"Validation condition {validation_condition} not found in loaded data")

        # Get training data
        Phi, Y_O = self.prepare_mimo_fir_data(condition)

        # Estimate parameters using all sensors
        z_all = np.ones(self.S)
        Phi_selected = self.apply_sensor_selection(Phi, z_all)
        Theta_hat = np.linalg.pinv(Phi_selected.T @ Phi_selected) @ Phi_selected.T @ Y_O

        # Get validation data
        validation_data = self.data[validation_condition]
        y_I_val = validation_data[:, self.channel_in]
        y_O_val = validation_data[:, self.channel_out]

        # Predict outputs using estimated parameters
        m_val = validation_data.shape[0]
        errors = []

        # Calculate total number of parameters
        h = (self.r + self.d + 1) * self.S

        for t in range(self.r + 1, m_val - self.d + 1):
            # Collect measurements for prediction
            phi_t = np.zeros(h)
            for s in range(self.S):
                col_idx_start = s * (self.r + self.d + 1)
                sensor_data = y_I_val[(t - self.r):(t + self.d + 1), s]
                phi_t[col_idx_start:(col_idx_start + self.r + self.d + 1)] = sensor_data[::-1]

            # Make prediction
            y_pred = phi_t @ Theta_hat

            # Compute error
            y_true = y_O_val[t - (self.r + 1), :]
            errors.append(y_true - y_pred)

        # Compute covariance matrix of errors
        errors = np.array(errors)
        error_mean = np.mean(errors, axis=0)
        Sigma_epsilon = np.zeros((self.output_dim, self.output_dim))

        for e in errors:
            e_centered = e - error_mean
            Sigma_epsilon += np.outer(e_centered, e_centered)

        Sigma_epsilon /= (len(errors) - 1)

        # Store the estimated covariance
        self.noise_covariance[condition] = Sigma_epsilon

        print(f"Estimated noise covariance for condition {condition}")
        return Sigma_epsilon

    def estimate_uncertainty_bound(self, condition, num_samples=10):
        """
        Estimate uncertainty bound beta for a condition.
        First checks if the bound exists in a cache file.
        If not, computes the bound and updates the cache.

        Parameters:
        -----------
        condition : str
            Name of the condition
        num_samples : int
            Number of samples to use for estimating bound

        Returns:
        --------
        beta : float
            Estimated uncertainty bound
        """
        # Define the cache file path
        cache_dir = "./result"
        cache_file = os.path.join(cache_dir, "uncertainty_bounds.json")

        # Create result directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Try to load existing cache
        bounds_cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    bounds_cache = json.load(f)
                print(f"Loaded uncertainty bounds cache from {cache_file}")
            except Exception as e:
                print(f"Error loading cache file: {e}")

        # Check if the bound for this condition is in the cache
        if condition in bounds_cache:
            beta = bounds_cache[condition]
            print(f"Found cached uncertainty bound for condition {condition}: {beta:.6f}")
            self.uncertainty_bounds[condition] = beta
            return beta

        # If not in cache, compute the bound
        if condition not in self.data:
            raise ValueError(f"Condition {condition} not found in loaded data")

        data = self.data[condition]
        m = data.shape[0]

        # Prepare nominal measurement matrix
        Phi_nominal, _ = self.prepare_mimo_fir_data(condition)

        # Estimate bound by sampling different segments of the data
        segment_size = m // (num_samples + 1)
        max_norm_ratio = 0

        for i in range(num_samples):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size

            # Create a subset of data
            subset_data = data[start_idx:end_idx, :]
            subset_condition = f"{condition}_subset_{i}"
            self.data[subset_condition] = subset_data

            # Prepare measurement matrix for this subset
            Phi_subset, _ = self.prepare_mimo_fir_data(subset_condition)

            # Compute delta
            min_rows = min(Phi_nominal.shape[0], Phi_subset.shape[0])
            delta = Phi_subset[:min_rows, :] - Phi_nominal[:min_rows, :]

            # Compute norm ratio
            norm_ratio = np.linalg.norm(delta, 2) / np.linalg.norm(Phi_nominal[:min_rows, :], 2)
            max_norm_ratio = max(max_norm_ratio, norm_ratio)

            # Clean up temporary data
            del self.data[subset_condition]

        # Set uncertainty bound
        beta = max_norm_ratio
        self.uncertainty_bounds[condition] = beta

        # Update the cache
        bounds_cache[condition] = beta
        try:
            with open(cache_file, 'w') as f:
                json.dump(bounds_cache, f, indent=4)
            print(f"Updated uncertainty bounds cache in {cache_file}")
        except Exception as e:
            print(f"Error updating cache file: {e}")

        print(f"Estimated uncertainty bound for condition {condition}: {beta:.6f}")
        return beta

    def apply_sensor_selection(self, Phi, z):
        """
        Apply sensor selection to measurement matrix.

        Parameters:
        -----------
        Phi : numpy.ndarray
            Measurement matrix
        z : cvxpy.Variable
            Sensor selection variable (S-dimensional)

        Returns:
        --------
        Phi_selected : cvxpy.Expression
            Measurement matrix after sensor selection
        """

        m = Phi.shape[0]  # Number of samples
        h = (self.r + self.d + 1) * self.S  # Total parameters
        block_size = self.r + self.d + 1

        # Instead of creating D matrix with z values directly,
        # we'll construct Phi_selected block by block
        Phi_blocks = []

        for s in range(self.S):
            # Extract the corresponding block of Phi for sensor s
            start_col = s * block_size
            end_col = (s + 1) * block_size
            Phi_block = Phi[:, start_col:end_col]

            # Multiply each block by corresponding z value
            Phi_blocks.append(z[s] * Phi_block)

        # Horizontally concatenate all blocks
        Phi_selected = cp.hstack(Phi_blocks)

        return Phi_selected

    def solve_robust_selection(self, conditions, p, relaxed=True):
        """
        Solve the robust sensor selection problem.

        Parameters:
        -----------
        conditions : list
            List of condition names to consider
        p : int
            Number of sensors to select
        relaxed : bool
            Whether to solve the relaxed (continuous) problem

        Returns:
        --------
        z_opt : numpy.ndarray
            Optimal sensor selection vector
        gamma_opt : float
            Optimal objective value
        """
        Q = len(conditions)  # Number of conditions
        h = (self.r + self.d + 1) * self.S  # Total parameters

        # Prepare data for all conditions
        condition_data = {}
        for q, condition in enumerate(conditions):
            Phi, Y_O = self.prepare_mimo_fir_data(condition)

            # Estimate uncertainty bound if not already done
            if condition not in self.uncertainty_bounds:
                self.estimate_uncertainty_bound(condition)

            # Estimate noise covariance if not already done
            if condition not in self.noise_covariance:
                try:
                    self.estimate_noise_covariance(condition)
                except ValueError:
                    # If validation data not available, use identity
                    print(f"No validation data for {condition}, using identity for noise covariance")
                    self.noise_covariance[condition] = np.eye(self.output_dim)

            condition_data[q] = {
                'Phi': Phi,
                'beta': self.uncertainty_bounds[condition],
                'Sigma_epsilon': self.noise_covariance[condition]
            }

        # Define optimization variables
        z = cp.Variable(self.S)  # Sensor selection vector
        gamma = cp.Variable()  # Minimum performance across conditions
        W = {}  # Auxiliary variables for each condition

        for q in range(Q):
            W[q] = cp.Variable((h, h), symmetric=True)

        # Define constraints
        constraints = [cp.sum(z) == p]  # Select exactly p sensors

        if relaxed:
            # Relaxed binary constraint
            constraints += [0 <= z, z <= 1]
        else:
            # Binary constraint (makes problem non-convex)
            constraints += [z >= 0, z <= 1, cp.mixed_integer(z, integer=True)]

        # Add robust SDP constraints for each condition
        for q in range(Q):
            Phi_q = condition_data[q]['Phi']
            beta_q = condition_data[q]['beta']
            Sigma_q = condition_data[q]['Sigma_epsilon']
            lambda_q = 1.0 / (beta_q ** 2)

            # Calculate Phi(z) using the current value of z
            Phi_z = self.apply_sensor_selection(Phi_q, z)

            # # Create a parameter for Phi(z) and set its value
            # Phi_z_param = cp.Parameter((Phi_q.shape[0], h))
            # Phi_z_param.value = Phi_z_value

            # Define the Schur complement constraint
            schur_matrix = cp.bmat([
                [W[q], Phi_z.T],
                [Phi_z, lambda_q * np.eye(Phi_q.shape[0])]
            ])

            # Constraint: Schur complement is PSD
            constraints.append(schur_matrix >> 0)

            # Constraint: minimum performance
            log_det_sigma = np.log(np.linalg.det(Sigma_q))
            constraints.append(6 * cp.log_det(W[q]) - h * log_det_sigma >= gamma)

        # Define objective
        objective = cp.Maximize(gamma)

        # Create problem
        problem = cp.Problem(objective, constraints)

        # Iterative approach for non-convex problem due to Phi(z) dependency
        z_current = np.ones(self.S) / self.S  # Start with uniform weights
        gamma_values = []
        max_iterations = 20
        tolerance = 1e-4

        # for iteration in range(max_iterations):
        #     print(f"Iteration {iteration + 1}/{max_iterations}")
        #
        #     # Update Phi(z) parameters based on current z
        #     # for q in range(Q):
        #     #     Phi_q = condition_data[q]['Phi']
        #     #     Phi_z_param = cp.Parameter((Phi_q.shape[0], h))
        #     #     Phi_z_current = self.apply_sensor_selection(Phi_q, z_current)
        #     #     Phi_z_param.value = Phi_z_current
        #
        #     # Solve the problem
        try:
            problem.solve(solver=cp.MOSEK, verbose=True)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Problem status: {problem.status}")
                # break

            # Get current solution
            z_new = z.value
            gamma_current = gamma.value
            gamma_values.append(gamma_current)

            # # Check convergence
            # if iteration > 0:
            #     z_change = np.linalg.norm(z_new - z_current)
            #     gamma_change = abs(gamma_values[-1] - gamma_values[-2])
            #     print(f"  z change: {z_change:.6f}, gamma change: {gamma_change:.6f}")
            #
            #     if z_change < tolerance and gamma_change < tolerance:
            #         print("Converged!")
            #         break

            # Update current solution
            z_current = z_new

        except cp.error.SolverError:
            print("Solver error occurred.")
            # break

        # Round solution if relaxed
        if relaxed:
            # Sort sensors by their values
            sorted_indices = np.argsort(z_current)[::-1]

            # Select top p sensors
            z_binary = np.zeros(self.S)
            z_binary[sorted_indices[:p]] = 1

            # Compute objective with rounded solution
            # (This would require evaluating the objective manually)

            z_opt = z_binary
        else:
            z_opt = z_current

        # Return optimal solution
        return z_opt, gamma_values[-1] if gamma_values else float('-inf')

    def evaluate_selection(self, z, condition):
        """
        Evaluate a sensor selection for a specific condition.

        Parameters:
        -----------
        z : numpy.ndarray
            Sensor selection vector
        condition : str
            Condition to evaluate on

        Returns:
        --------
        mse : float
            Mean squared error of estimation
        """
        if condition not in self.data:
            raise ValueError(f"Condition {condition} not found in loaded data")

        # Prepare data
        Phi, Y_O = self.prepare_mimo_fir_data(condition)
        Phi_selected = self.apply_sensor_selection(Phi, z)

        # Compute least squares estimate
        Theta_hat = np.linalg.pinv(Phi_selected) @ Y_O

        # Compute predictions
        Y_pred = Phi_selected @ Theta_hat

        # Compute MSE
        mse = np.mean((Y_O - Y_pred) ** 2)

        return mse

    def plot_selection_results(self, z_opt, conditions):
        """
        Plot the results of sensor selection.

        Parameters:
        -----------
        z_opt : numpy.ndarray
            Optimal sensor selection vector
        conditions : list
            List of conditions used in optimization
        """
        # Plot selected sensors
        plt.figure(figsize=(12, 6))

        # First subplot: selected sensors
        plt.subplot(1, 2, 1)
        selected_indices = np.where(z_opt > 0.5)[0]
        channel_indices = [self.channel_in[i] for i in selected_indices]

        plt.bar(range(len(selected_indices)), np.ones(len(selected_indices)))
        plt.xticks(range(len(selected_indices)), channel_indices, rotation=90)
        plt.xlabel('Selected Channel Indices')
        plt.ylabel('Selection Status')
        plt.title(f'Selected {len(selected_indices)} Sensors')

        # Second subplot: performance across conditions
        plt.subplot(1, 2, 2)
        mse_values = []

        for condition in conditions:
            mse = self.evaluate_selection(z_opt, condition)
            mse_values.append(mse)

        plt.bar(range(len(conditions)), mse_values)
        plt.xticks(range(len(conditions)), conditions, rotation=45)
        plt.xlabel('Operating Condition')
        plt.ylabel('Mean Squared Error')
        plt.title('Performance Across Conditions')

        plt.tight_layout()
        plt.show()

    def compare_with_baseline(self, z_opt, conditions, num_baselines=5):
        """
        Compare optimal selection with random baselines.

        Parameters:
        -----------
        z_opt : numpy.ndarray
            Optimal sensor selection vector
        conditions : list
            List of conditions to evaluate on
        num_baselines : int
            Number of random baselines to compare with
        """
        p = int(np.sum(z_opt))  # Number of selected sensors

        # Evaluate optimal selection
        opt_mse = {}
        for condition in conditions:
            opt_mse[condition] = self.evaluate_selection(z_opt, condition)

        # Generate random baselines
        baseline_mse = {condition: [] for condition in conditions}

        for i in range(num_baselines):
            # Random selection of p sensors
            z_random = np.zeros(self.S)
            random_indices = np.random.choice(self.S, p, replace=False)
            z_random[random_indices] = 1

            # Evaluate on each condition
            for condition in conditions:
                mse = self.evaluate_selection(z_random, condition)
                baseline_mse[condition].append(mse)

        # Plot comparison
        plt.figure(figsize=(12, 6))

        for i, condition in enumerate(conditions):
            plt.subplot(1, len(conditions), i + 1)

            # Baseline distribution
            plt.hist(baseline_mse[condition], bins=10, alpha=0.7, label='Random')

            # Optimal selection
            plt.axvline(opt_mse[condition], color='r', linestyle='dashed', linewidth=2, label='Optimal')

            plt.title(f'Condition: {condition}')
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Frequency')

            if i == 0:
                plt.legend()

        plt.tight_layout()
        plt.show()

        # Print improvement statistics
        print("\nImprovement Statistics:")
        print("----------------------")

        for condition in conditions:
            baseline_mean = np.mean(baseline_mse[condition])
            improvement = (baseline_mean - opt_mse[condition]) / baseline_mean * 100
            print(f"Condition {condition}: {improvement:.2f}% improvement over random baseline")


def load_data(rss, data_folder, conditions):
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
            condition_name = f"{base_condition}_{len(training_files) + 1:02d}"
            file_path = os.path.join(data_folder, validation_file)
            rss.load_data(file_path, condition_name)
            loaded_conditions.append(condition_name)
            print(f"Loaded validation data for {base_condition} from {validation_file}")

    return loaded_conditions


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
    loaded_conditions = load_data(rss, data_folder, conditions)

    if len(loaded_conditions) == 0:
        raise ValueError("No valid conditions could be loaded. Check data folder and condition names.")

    # Solve the robust sensor selection problem
    print(f"Solving robust sensor selection for {p} sensors across {len(loaded_conditions)} conditions")
    z_opt, gamma_opt = rss.solve_robust_selection(loaded_conditions, p, relaxed=True)
    print("Optimization complete!")
    print("z_opt:", z_opt)
    # Get selected channels
    selected_indices = np.where(z_opt > 0.5)[0]
    selected_channels = [channel_in[i] for i in selected_indices]

    print(f"Selected {len(selected_indices)} out of {len(channel_in)} sensors:")
    for i, idx in enumerate(selected_indices):
        print(f"  {i + 1}: Channel {channel_in[idx]}")

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
    data_folder = '../Data/matdata/'  # Adjust to your data path

    # Use base condition names (without specific numbers)
    # The function will automatically find all matching files and use appropriate training/validation split
    # conditions = ['BC30', 'BC40', 'BC50']
    conditions = ['BC30']
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
        loaded_conditions = load_data(rss, data_folder, conditions)

        # Load and evaluate additional conditions
        additional_loaded = load_data(rss, data_folder, additional_conditions)

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
            f.write(f"{i + 1}, {channel}\n")
    print(f"Detailed selected sensors saved to 'selected_sensors_detailed.csv'")

    return z_opt, selected_channels


if __name__ == "__main__":
    main()
