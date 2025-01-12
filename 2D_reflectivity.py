import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_data(Epump_file, Eref_file):
    """
    Reads the E_pump and E_ref CSV files and extracts the data.
    """
    E_Pump_data = pd.read_csv(Epump_file, skiprows=1)
    E_Ref_data = pd.read_csv(Eref_file, skiprows=1)
    
    time_axis = E_Pump_data.iloc[:, 0].values  # Time is in the first column
    E_Pump = E_Pump_data.iloc[:, 1:].values    # Pump data in the remaining columns
    E_Ref = E_Ref_data.iloc[:, 1:].values      # Reference data in the remaining columns

    return E_Ref, E_Pump, time_axis

def calculate_differential_reflectivity(E_Ref, E_Pump, time_axis):
    """
    Calculates differential reflectivity as a function of frequency and pump delay.
    """
    # Compute the frequency axis
    dt = time_axis[1] - time_axis[0]  # Time step
    freqs = np.fft.fftfreq(len(time_axis), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Keep only positive frequencies

    # Fourier transform the fields
    E_Ref_fft = np.fft.fft(E_Ref, axis=0)
    E_Pump_fft = np.fft.fft(E_Pump, axis=0)

    # Compute differential reflectivity
    delta_E = E_Pump_fft[:len(freqs), :] - E_Ref_fft[:len(freqs), :]
    E_Ref_fft = E_Ref_fft[:len(freqs), :]
    dR = delta_E / E_Ref_fft
    return np.real(dR), freqs

def plot_reflectivity_heatmap(reflectivity, freqs, pump_delays, output_file, max_freq=10):
    """
    Plots a 2D heat map of reflectivity with frequency on the x-axis.
    """

    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask] 
    reflectivity = reflectivity[freq_mask, :]

    plt.figure(figsize=(12, 8))
    extent = [freqs[0], freqs[-1], pump_delays[0], pump_delays[-1]]
    plt.imshow(reflectivity.T, aspect="auto", extent=extent, origin="lower", cmap="viridis")
    plt.colorbar(label="Differential Reflectivity")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Pump Delay (ps)")
    plt.title("Differential Reflectivity Heatmap")
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    # Input files
    Epump_file = "processed_data_temp200/E_pump.csv"
    Eref_file = "processed_data_temp200/E_ref.csv"
    output_folder = "reflection_analysis_temp200"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    E_Ref, E_Pump, time_axis = get_data(Epump_file, Eref_file)
   
    # Calculate differential reflectivity
    dreflectivity, freqs = calculate_differential_reflectivity(E_Ref, E_Pump, time_axis)

    # Define pump delays
    initial_delay = -0.3  # Start pump delay in ps
    final_delay = 2.0    # End pump delay in ps
    step_size = 0.05     # Step size in ps
    additional_delays = "110"  # Additional delays, comma-separated

    # Parse additional delays and combine with regular delays
    additional_delays = [float(x) for x in additional_delays.split(",")] if additional_delays else []
    pump_delays = list(np.arange(initial_delay, final_delay + step_size, step_size)) + additional_delays
    pump_delays.sort()

    # Plot and save heatmap with frequency on x-axis
    output_file = os.path.join(output_folder, "reflectivity_heatmap.png")
    plot_reflectivity_heatmap(dreflectivity, freqs, pump_delays, output_file, max_freq=10)
