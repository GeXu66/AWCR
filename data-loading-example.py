"""
Data loading and inspection script for wheel force estimation.
This script helps to understand the data structure and verify proper loading.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import argparse


def inspect_mat_file(file_path):
    """
    Inspect a MATLAB .mat file and print its contents and structure.

    Parameters:
    -----------
    file_path : str
        Path to the MATLAB file
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None

    try:
        mat_data = sio.loadmat(file_path)

        print(f"\nInspecting file: {file_path}")
        print("=" * 50)

        # Print keys in the file
        print("Keys in the file:")
        for key in mat_data:
            if not key.startswith('__'):  # Skip MATLAB metadata
                if isinstance(mat_data[key], np.ndarray):
                    print(f"  {key}: array of shape {mat_data[key].shape}, dtype {mat_data[key].dtype}")
                else:
                    print(f"  {key}: {type(mat_data[key])}")

        # Find main data array (usually the largest one)
        main_data = None
        max_size = 0
        main_key = None

        for key in mat_data:
            if not key.startswith('__'):
                if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) >= 2:
                    size = np.prod(mat_data[key].shape)
                    if size > max_size:
                        max_size = size
                        main_data = mat_data[key]
                        main_key = key

        if main_data is not None:
            print(f"\nMain data found under key '{main_key}' with shape {main_data.shape}")

            # Analyze data statistics
            print("\nData statistics:")
            if len(main_data.shape) == 2:
                print(f"  Number of samples: {main_data.shape[0]}")
                print(f"  Number of channels: {main_data.shape[1]}")

                # Check for missing values
                missing_count = np.sum(np.isnan(main_data))
                if missing_count > 0:
                    print(f"  Warning: {missing_count} missing values found!")

                # Show basic statistics for a few channels
                print("\nSample channel statistics:")
                channels_to_show = min(5, main_data.shape[1])
                for i in range(channels_to_show):
                    channel_data = main_data[:, i]
                    print(f"  Channel {i}: min={np.min(channel_data):.4f}, max={np.max(channel_data):.4f}, mean={np.mean(channel_data):.4f}, std={np.std(channel_data):.4f}")

                # Check for constant channels (no variation)
                constant_channels = []
                for i in range(main_data.shape[1]):
                    if np.std(main_data[:, i]) < 1e-6:
                        constant_channels.append(i)

                if constant_channels:
                    print(f"\n  Found {len(constant_channels)} constant channels (no variation):")
                    print(f"  Channels: {constant_channels[:10]}{'...' if len(constant_channels) > 10 else ''}")

                return main_data
            else:
                print("  Data has more than 2 dimensions. Basic inspection only.")
                return main_data
        else:
            print("\nNo suitable main data array found.")
            return None

    except Exception as e:
        print(f"Error inspecting file {file_path}: {e}")
        return None


def plot_channels(data, channels, title=None, max_points=2000):
    """
    Plot specified channels from the data.

    Parameters:
    -----------
    data : numpy.ndarray
        Data array with shape (samples, channels)
    channels : list
        List of channel indices to plot
    title : str, optional
        Plot title
    max_points : int, optional
        Maximum number of points to plot (for performance)
    """
    if data is None or len(data.shape) != 2:
        print("Invalid data format for plotting.")
        return

    # Subsample if needed
    if data.shape[0] > max_points:
        indices = np.linspace(0, data.shape[0] - 1, max_points, dtype=int)
        plot_data = data[indices, :]
    else:
        plot_data = data

    plt.figure(figsize=(12, 8))

    # Get a colormap
    cmap = get_cmap('tab10')

    for i, channel in enumerate(channels):
        if channel < data.shape[1]:
            plt.plot(plot_data[:, channel], label=f"Channel {channel}", color=cmap(i % 10))
        else:
            print(f"Channel {channel} is out of range.")

    plt.legend()
    plt.grid(True)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_input_output_channels(data, channel_in, channel_out, title=None, max_channels=5):
    """
    Plot sample input and output channels.

    Parameters:
    -----------
    data : numpy.ndarray
        Data array with shape (samples, channels)
    channel_in : list
        List of input channel indices
    channel_out : list
        List of output channel indices
    title : str, optional
        Plot title
    max_channels : int, optional
        Maximum number of channels to plot in each category
    """
    if data is None or len(data.shape) != 2:
        print("Invalid data format for plotting.")
        return

    plt.figure(figsize=(15, 10))

    # Plot input channels
    plt.subplot(2, 1, 1)
    in_cmap = get_cmap('Blues')
    in_channels = channel_in[:max_channels]

    for i, channel in enumerate(in_channels):
        if channel < data.shape[1]:
            # Normalize to make comparison easier
            channel_data = data[:, channel]
            normalized = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
            plt.plot(normalized, label=f"In Ch {channel}", color=in_cmap(0.3 + 0.7 * (i + 1) / len(in_channels)))

    plt.legend()
    plt.grid(True)
    plt.title("Sample Input Channels (Normalized)")

    # Plot output channels
    plt.subplot(2, 1, 2)
    out_cmap = get_cmap('Reds')

    for i, channel in enumerate(channel_out):
        if channel < data.shape[1]:
            channel_data = data[:, channel]
            normalized = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
            plt.plot(normalized, label=f"Out Ch {channel} ({['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'][i] if i < 6 else ''})",
                     color=out_cmap(0.3 + 0.7 * (i + 1) / len(channel_out)))

    plt.legend()
    plt.grid(True)
    plt.title("Output Channels (Normalized)")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inspect wheel force MAT files')
    parser.add_argument('--file', type=str, help='Path to MAT file to inspect')
    parser.add_argument('--folder', type=str, default='./', help='Folder containing MAT files')
    parser.add_argument('--pattern', type=str, default='BC', help='File name pattern to match')
    args = parser.parse_args()

    # Define input and output channels according to the problem specification
    channel_in = list(range(0, 32)) + list(range(56, 139))
    channel_in = [x for x in channel_in if x not in [57, 65, 76, 79]]
    channel_out = [32, 33, 34, 35, 36, 37]

    # Inspect a specific file if provided
    if args.file:
        if os.path.exists(args.file):
            data = inspect_mat_file(args.file)
            if data is not None:
                print("\nPlotting sample channels...")
                plot_input_output_channels(data, channel_in, channel_out,
                                           title=f"Channels from {os.path.basename(args.file)}")
        else:
            print(f"File {args.file} does not exist.")

    # Inspect all matching files in the folder
    else:
        files = []
        for file in os.listdir(args.folder):
            if file.endswith('.mat') and args.pattern in file:
                files.append(file)

        files.sort()

        if not files:
            print(f"No .mat files matching '{args.pattern}' found in folder {args.folder}")
            return

        print(f"Found {len(files)} matching files:")
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")

        # Let user choose a file
        while True:
            try:
                choice = input("\nEnter the number of the file to inspect (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    file_path = os.path.join(args.folder, files[idx])
                    data = inspect_mat_file(file_path)
                    if data is not None:
                        print("\nPlotting sample channels...")
                        plot_input_output_channels(data, channel_in, channel_out,
                                                   title=f"Channels from {files[idx]}")
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number or 'q'.")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
