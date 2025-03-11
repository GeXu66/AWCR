# Cross-Coupled Multiplicative-Robust Sensor Selection

This implementation provides a comprehensive solution for the robust sensor selection problem in wheel force estimation, as described in the paper. The code implements the theoretical framework including the MIMO-FIR model, ellipsoidal uncertainty characterization, and multiplicative-robust sensor selection using semidefinite programming (SDP).

## Problem Description

The sensor selection problem aims to select a subset of p sensors from a total of S available sensors to estimate wheel forces and moments under various operating conditions. The algorithm ensures robust performance by accounting for:

1. Multiple operating conditions with different dynamics
2. Cross-coupling between the six wheel force/moment components
3. Multiplicative uncertainties in measurements
4. Optimization of the worst-case performance across all conditions

## Code Structure

The implementation consists of the following main components:

1. **RobustSensorSelection Class**: The core class implementing the theoretical framework
2. **Run Script**: A utility script to demonstrate the usage of the implementation
3. **Data Loading Utilities**: Functions to preprocess and load MATLAB data files

## Installation and Dependencies

The implementation requires the following Python packages:

```bash
pip install numpy scipy cvxpy matplotlib tqdm
```

## Usage

### 1. Data Preparation

Prepare your MATLAB (.mat) data files with the following structure:
- Each file should represent one operating condition
- Each file should contain time-series data with channels corresponding to sensor measurements
- Input channels and output channels should be specified according to your setup

### 2. Basic Usage

```python
# Import the RobustSensorSelection class
from robust_sensor_selection import RobustSensorSelection

# Define input and output channels
channel_in = list(range(0, 32)) + list(range(56, 139))
channel_in = [x for x in channel_in if x not in [57, 65, 76, 79]]
channel_out = [32, 33, 34, 35, 36, 37]

# Create selection object
rss = RobustSensorSelection(r=3, d=3)
rss.set_channels(channel_in, channel_out)

# Load data for conditions
conditions = ['BC30_01', 'BC40_01', 'BC50_01']
data_dir = './'  # Adjust path as needed

for condition in conditions:
    file_path = os.path.join(data_dir, f'{condition}.mat')
    rss.load_data(file_path, condition)

# Solve the robust sensor selection problem
p = 20  # Number of sensors to select
z_opt, gamma_opt = rss.solve_robust_selection(conditions, p, relaxed=True)

# Analyze and visualize results
rss.plot_selection_results(z_opt, conditions)
rss.compare_with_baseline(z_opt, conditions, num_baselines=10)
```

### 3. Running the Complete Example

To run the complete example with the provided data:

```bash
python run_sensor_selection.py
```

This will:
1. Load the specified MATLAB data files
2. Run the robust sensor selection algorithm
3. Visualize the results
4. Save the selected sensors to a file

## Implementation Details

### MIMO-FIR Model

The implementation uses a Multiple-Input Multiple-Output Finite Impulse Response (MIMO-FIR) model as described in the paper. For each operating condition, the model relates input measurements to output forces/moments with both past and future time steps.

### Uncertainty Modeling

The code implements two key aspects of uncertainty:
1. **Noise Covariance Matrix**: Capturing the cross-coupling between different force/moment components
2. **Multiplicative Uncertainty**: Modeling the variations in the measurement process across different operating conditions

### Optimization Approach

The sensor selection is formulated as a semidefinite programming (SDP) problem using CVXPY. Key features:
1. Convex relaxation of binary variables
2. Min-max optimization across multiple operating conditions
3. Robust formulation against multiplicative uncertainties
4. Efficient iterative algorithm for solving the non-convex problem

## Convexity Analysis

The original problem with binary sensor selection variables is non-convex. The implementation uses a convex relaxation by:
1. Relaxing binary constraints to continuous variables between 0 and 1
2. Using log-determinant objective which is concave
3. Applying Schur complement to transform uncertainty constraints into linear matrix inequalities

The resulting SDP problem is convex and can be efficiently solved with standard solvers.

## Extending the Implementation

To use this implementation with different data or to extend its capabilities:

1. **Different Data Format**: Modify the `load_data` method to support your data format
2. **Additional Uncertainty Models**: Extend the `estimate_uncertainty_bound` method
3. **Alternative Optimization Criteria**: Modify the objective function in `solve_robust_selection`
4. **Performance Metrics**: Add custom evaluation metrics in the `evaluate_selection` method

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: Reduce the data size or use data subsampling
2. **Solver Errors**: Try different solver parameters or alternative solvers in CVXPY
3. **Convergence Issues**: Increase the maximum iterations or adjust the tolerance in the iterative algorithm

## Citation

If using this implementation in your research, please cite the original paper.

## License

This implementation is provided for research and educational purposes only.