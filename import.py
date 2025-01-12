import os
import numpy as np
import pandas as pd
from pathlib import Path

def import_files():
    filedir = Path(os.getcwd())  # Current working directory
    filenames = get_file_list(filedir)  # List of files to open

    if not filenames:
        print("No files found.")
        return

    open_files(filedir, filenames)

def get_file_list(filedir):
    """Returns a list with the names of the files in the filedir folder."""
    dirfiles = sorted(
        filedir.glob("*Average.*"), key=os.path.getmtime
    )  
    return [file.name for file in dirfiles]

def open_files(filedir, filenames):
    first_file = filedir / filenames[0]
    data = pd.read_csv(first_file, delim_whitespace=True, header=None).values

    m = data.shape[0]  
    n = len(filenames) 

    # Initialize arrays
    E1_2D = np.zeros((m, n + 1))
    E2_2D = np.zeros((m, n + 1))

    # Copy time axis
    E1_2D[:, 0] = data[:, 0]
    E2_2D[:, 0] = data[:, 0]

    # Copy first file's data
    E1_2D[:, 1] = data[:, 1]
    E2_2D[:, 1] = data[:, 2]

    # Write Pumpindex.dat
    np.savetxt("Pumpindex.dat", np.arange(1, n + 1), fmt="%d", newline="\n")

    # Write PumpTimes.dat if it doesn't exist
    if not os.path.exists("PumpTimes.dat"):
        pump_times = np.arange(-3, n - 3)
        np.savetxt("PumpTimes.dat", pump_times, fmt="%d", newline="\n")

    # Process remaining files
    for i, filename in enumerate(filenames[1:], start=2):
        file = filedir / filename
        data = pd.read_csv(file, delim_whitespace=True, header=None).values

        E1_2D[:, i] = data[:, 1]
        E2_2D[:, i] = data[:, 2]

    # Save E1.dat and E2.dat
    np.savetxt("E1.dat", E1_2D, delimiter="\t", fmt="%.6e")
    np.savetxt("E2.dat", E2_2D, delimiter="\t", fmt="%.6e")

# Run the script
if __name__ == "__main__":
    import_files()