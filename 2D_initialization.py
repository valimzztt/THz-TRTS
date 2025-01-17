import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import re

def process_data(data_folder, output_folder):
    """
    Process files in the data folder, extract pump probe delays, and save 2D CSV files
    for both lock-in amplifiers.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get list of .d24 files in the folder
    data_folder = Path(data_folder)
    files = [data_folder / f for f in os.listdir(data_folder) if f.endswith('Average.d24')]
    def extract_date_from_filename(filename):
            try:
                date_str = ''.join(str(filename).split(' ')[1:3])
                return datetime.strptime(date_str, '%b%d%H%M')
            except (IndexError, ValueError) as e:
                print(f"Error processing file {filename}: {e}")
                return None
   
    # files.sort(key=lambda f: extract_date_from_filename(f) or datetime.min)
    def extract_pump_delay_number(filepath):
        """
        Extracts the pump delay number from the full file path.
        Returns the number as an integer or None if no number is found.
        """
        # Extract filename from the full path
        filename = str(filepath).split('/')[-1]  # Handle Unix-like paths

        match = re.search(r'pump_delay_(\d+)', filename)
        if match:
            return int(match.group(1))  # Extract and convert to integer
        return None  # Return None if no match is found

    # Sort files based on the extracted pump delay number
    # Use this one for T=200K
    # files.sort(key=lambda f: extract_pump_delay_number(f) or 0)  
    # Use this one for Roomtemp
    files.sort(key=lambda f: extract_date_from_filename(f) or datetime.min)
    print(files[1:5])
    try:
        """ start_delay = float(input("Enter the starting pump delay (in ps): "))
        stop_delay = float(input("Enter the stopping pump delay (in ps): "))
        step_size = float(input("Enter the step size for pump delays (in ps): "))
        additional_delays = input("Enter additional delays (comma-separated, in ps), or press Enter to skip: ") """
        start_delay = -0.4
        stop_delay = 2
        step_size = 0.05
        additional_delays = "110"
        # Generate pump delays using linspace and append additional delays
        pump_delays = list(np.round(np.arange(start_delay, stop_delay + step_size, step_size), 2))
        pump_delays = [round(delay, 2) for delay in pump_delays]
        pump_delays.sort()
        if additional_delays:
            pump_delays.extend([float(x) for x in additional_delays.split(",")])
        if len(pump_delays) != len(files):
            print(len)
            raise ValueError("Number of delays does not match the number of files, check carefully.")

    except ValueError as e:
        print(f"Error: {e}")
        return

    time_axis = None
    lockin1_matrix = []
    lockin2_matrix = []

    for file in files:
        data = pd.read_csv(file, delim_whitespace=True, skiprows=1, header=None, names=["Time (ps)", "LockIn1", "LockIn2"])

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
    # Compute E_ref and E_pump
    E_ref = lockin1_matrix[:, 1:] - lockin2_matrix[:, 1:]
    E_pump = lockin1_matrix[:, 1:] + lockin2_matrix[:, 1:]

    # Save E_ref and E_pump as CSV files
    E_ref_file = Path(output_folder) / "E_ref.csv"
    E_pump_file = Path(output_folder) / "E_pump.csv"

    np.savetxt(E_ref_file, np.column_stack([time_axis] + [E_ref[:, i] for i in range(E_ref.shape[1])]), delimiter=",", header="Time (ps)," + ",".join(map(str, pump_delays)), comments="", fmt="%.6f")
    np.savetxt(E_pump_file, np.column_stack([time_axis] + [E_pump[:, i] for i in range(E_pump.shape[1])]), delimiter=",", header="Time (ps)," + ",".join(map(str, pump_delays)), comments="", fmt="%.6f")

    print(f"E_ref data saved to: {E_ref_file}")
    print(f"E_pump data saved to: {E_pump_file}")
    os.makedirs(Path(output_folder+r"\LockIn_Plots") , exist_ok=True)
    os.makedirs(Path(output_folder+r"\E_Plots") , exist_ok=True)
    # Plot the data
    for i, delay in enumerate(pump_delays):
        plt.figure()
        # Subtract baselines
        lockin1_matrix[:, i + 1] -= np.mean(lockin1_matrix[:, i + 1], axis=0)
        lockin2_matrix[:, i + 1] -= np.mean(lockin2_matrix[:, i + 1], axis=0)
        plt.plot(time_axis, lockin1_matrix[:, i + 1], label=f"LockIn1 - Pump Delay {delay} ps")
        plt.plot(time_axis, lockin2_matrix[:, i + 1], label=f"LockIn2 - Pump Delay {delay} ps")
        plt.xlabel("Time (ps)")
        plt.ylabel("Signal")
        plt.title(f"Pump Delay {delay:.2f} ps")
        plt.legend()
        plt.grid()
        plt.savefig(Path(output_folder+r"\LockIn_Plots") / f"Plot_Pump_Delay_{delay:.2f}.png")
        plt.close()

        plt.figure()
        # Subtract baselines
        E_ref[:, i + 1] -= np.mean(E_ref[:, i + 1], axis=0)
        E_pump[:, i + 1] -= np.mean(E_pump[:, i + 1], axis=0)
        plt.plot(time_axis, E_ref[:, i + 1], label=f"E_ref - Pump Delay {delay} ps")
        plt.plot(time_axis, E_pump[:, i + 1], label=f"E_pump - Pump Delay {delay} ps")
        plt.xlabel("Time (ps)")
        plt.ylabel("Signal")
        plt.title(f"Pump Delay {delay:.2f} ps")
        plt.legend()
        plt.grid()
        plt.savefig(Path(output_folder+r"\E_Plots") / f"Plot_Pump_Delay_{delay:.2f}.png")
        plt.close()

if __name__ == "__main__":
    data_folder = "Dec16_New2DScan_RoomTemp" 
    output_folder = "processed_data_roomtemp_new"
    process_data(data_folder, output_folder)

