import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def process_data(data_folder, output_folder):
    """
    Process files in the data folder, extract pump probe delays, and save 2D CSV files
    for both lock-in amplifiers.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of .d24 files in the folder
    data_folder = Path(data_folder)
    files = sorted([f for f in data_folder.glob("*.d24")], key=os.path.getmtime)

    if not files:
        print("No .d24 files found in the folder.")
        return

    # Prompt the user for pump probe delays
    try:
        start_delay = float(input("Enter the starting pump delay (in ps): "))
        stop_delay = float(input("Enter the stopping pump delay (in ps): "))
        step_size = float(input("Enter the step size for pump delays (in ps): "))
        additional_delays = input("Enter additional delays (comma-separated, in ps), or press Enter to skip: ")

        # Generate pump delays using linspace and append additional delays
        pump_delays = list(np.arange(start_delay, stop_delay + step_size, step_size))
        if additional_delays:
            pump_delays.extend([float(x) for x in additional_delays.split(",")])
        print(len(pump_delays))
        print(len(files))
        if len(pump_delays) != len(files):
            raise ValueError("Number of delays does not match the number of files.")

    except ValueError as e:
        print(f"Error: {e}")
        return

    time_axis = None
    lockin1_matrix = []
    lockin2_matrix = []

    for file in files:
        # Read the file
        data = pd.read_csv(file, delim_whitespace=True, header=None, names=["Time (ps)", "LockIn1", "LockIn2", "Column4", "Column5"])


        if time_axis is None:
            time_axis = data["Time (ps)"].values  # Set the time axis from the first file

        lockin1_matrix.append(data["LockIn1"].values)
        lockin2_matrix.append(data["LockIn2"].values)

    # Convert matrices to numpy arrays and add time axis as the first column
    lockin1_matrix = np.column_stack([time_axis] + lockin1_matrix)
    lockin2_matrix = np.column_stack([time_axis] + lockin2_matrix)

    # Save matrices as CSV files
    lockin1_file = Path(output_folder) / "LockIn1.csv"
    lockin2_file = Path(output_folder) / "LockIn2.csv"

    np.savetxt(lockin1_file, lockin1_matrix, delimiter=",", header="Time (ps)," + ",".join(map(str, pump_delays)), comments="", fmt="%.6f")
    np.savetxt(lockin2_file, lockin2_matrix, delimiter=",", header="Time (ps)," + ",".join(map(str, pump_delays)), comments="", fmt="%.6f")

    print(f"LockIn1 data saved to: {lockin1_file}")
    print(f"LockIn2 data saved to: {lockin2_file}")

    # Plot the data
    for i, delay in enumerate(pump_delays):
        plt.figure()
        plt.plot(time_axis, lockin1_matrix[:, i + 1], label=f"LockIn1 - Pump Delay {delay} ps")
        plt.plot(time_axis, lockin2_matrix[:, i + 1], label=f"LockIn2 - Pump Delay {delay} ps")
        plt.xlabel("Time (ps)")
        plt.ylabel("Signal")
        plt.title(f"Pump Delay {delay:.2f} ps")
        plt.legend()
        plt.grid()
        plt.savefig(Path(output_folder) / f"Plot_Pump_Delay_{delay:.2f}.png")
        plt.close()

if __name__ == "__main__":
    data_folder = "Dec12_2D_Temp200_AveragedData"  # Folder containing .d24 files
    output_folder = "processed_data"  # Output folder for CSV files and plots
    process_data(data_folder, output_folder)

