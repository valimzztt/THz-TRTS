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

def process_data(E_Ref, E_Pump, time_axis):
    E_Ref -= np.mean(E_Ref)
    E_Pump -= np.mean(E_Pump)
    global processed_E_Ref, processed_E_Pump
    global E_Ref_windowed, E_Pump_windowed

    def create_centered_hann(data_length, window_center, window_width):
        hann_window = np.hanning(window_width)
        padded_window = np.zeros(data_length)
        start = max(0, window_center - window_width // 2)
        end = min(data_length, start + window_width)
        padded_window[start:end] = hann_window[:(end - start)]
        return padded_window

    hann_window_center = 100
    hann_window_width = 400
    ref_hann_window = create_centered_hann(len(E_Ref), hann_window_center, hann_window_width)
    pump_hann_window = create_centered_hann(len(E_Pump), hann_window_center, hann_window_width)
    E_Ref_windowed = E_Ref * ref_hann_window[:, np.newaxis]
    E_Pump_windowed = E_Pump * pump_hann_window[:, np.newaxis]

    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]
    E_Ref_FFT = np.fft.fft(E_Ref)[:len(freqs)]
    E_Pump_FFT = np.fft.fft(E_Pump)[:len(freqs)]
    E_Ref_windowed_FFT = np.fft.fft(E_Ref_windowed)[:len(freqs)]
    E_Pump_windowed_FFT = np.fft.fft(E_Pump_windowed)[:len(freqs)]

    return E_Ref_windowed, E_Pump_windowed, time_axis

def plot_fft_pump_time_axis(reflectivity, pump_delays, freqs, output_file, csv_output_file, max_freq=10, max_fft=5):
    """
    Plots a 2D FFT map along the pump delay axis and frequency axis.

    Parameters:
    - reflectivity: 2D array of reflectivity values.
    - pump_delays: List of pump delays.
    - freqs: Frequency array.
    - output_file: File path to save the FFT heatmap image.
    - csv_output_file: File path to save the FFT data as CSV.
    - max_freq: Maximum frequency to display in the FFT heatmap.
    - max_fft: Maximum FFT amplitude to display in the heatmap.

    Returns:
    - fft_map_df: DataFrame of the FFT map.
    """
    # Perform FFT along the pump delay axis
    fft_reflectivity = np.fft.fft(reflectivity, axis=1)
    fft_amplitude = np.abs(fft_reflectivity)
    fft_freqs = np.fft.fftfreq(len(pump_delays), d=np.abs(pump_delays[1] - pump_delays[0]))

    # Keep only the positive frequencies
    fft_amplitude = fft_amplitude[:, :len(fft_freqs) // 2]
    fft_freqs = fft_freqs[:len(fft_freqs) // 2]

    # Mask frequencies and clip amplitudes
    freq_mask = freqs <= max_freq
    fft_amplitude_clipped = np.clip(fft_amplitude[freq_mask, :], None, max_fft)

    # Save the FFT map data as a CSV file
    fft_map_df = pd.DataFrame(
        data=fft_amplitude_clipped.T,
        index=np.round(fft_freqs, 2),
        columns=np.round(freqs[freq_mask], 2),
    )
    fft_map_df.index.name = "Pump Delay Frequency (Hz)"
    fft_map_df.columns.name = "Reflectivity Frequency (THz)"
    fft_map_df.to_csv(csv_output_file)

    # Plot the FFT heatmap
    plt.figure(figsize=(12, 8))
    extent = [freqs[freq_mask][0], freqs[freq_mask][-1], fft_freqs[0], fft_freqs[-1]]
    plt.imshow(
        fft_amplitude_clipped.T,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmax=max_fft,  # Limit the maximum FFT amplitude
    )
    plt.colorbar(label="FFT Amplitude (Clipped at {})".format(max_fft))
    plt.xlabel("Reflectivity Frequency (THz)")
    plt.ylabel("Pump Delay Frequency (Hz)")
    plt.title("FFT Heatmap")
    plt.savefig(output_file)
    plt.show()

    print(f"FFT heatmap plot saved to {output_file}")
    print(f"FFT map data saved to {csv_output_file}")

    return fft_map_df


def plot_reflectivity_heatmap_clipped(reflectivity, freqs, pump_delays, output_file, csv_output_file, max_freq=10, max_reflectivity=5):
    """
    Plots a 2D heat map of reflectivity with frequency on the x-axis, limiting reflectivity to a maximum value.
    
    Parameters:
    - reflectivity: 2D array of reflectivity values.
    - freqs: Array of frequency values.
    - pump_delays: List of pump delays.
    - output_file: File path to save the heatmap image.
    - csv_output_file: File path to save the reflectivity data as CSV.
    - max_freq: Maximum frequency to display in the heatmap.
    - max_reflectivity: Maximum reflectivity to display in the heatmap.
    
    Returns:
    - heatmap_df: DataFrame of the heatmap.
    """
    # Mask frequencies above the max frequency
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    reflectivity = reflectivity[freq_mask, :]

    # Clip reflectivity values to max_reflectivity
    reflectivity_clipped = np.clip(reflectivity, None, max_reflectivity)

    # Save the heatmap data as a CSV file
    heatmap_df = pd.DataFrame(
        data=reflectivity.T,
        index=np.round(pump_delays, 2),
        columns=np.round(freqs, 2),
    )
    heatmap_df.index.name = "Pump Delay (ps)"
    heatmap_df.columns.name = "Frequency (THz)"
    heatmap_df.to_csv(csv_output_file)

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    extent = [freqs[0], freqs[-1], pump_delays[0], pump_delays[-1]]
    plt.imshow(
        reflectivity_clipped.T,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmax=max_reflectivity,  # Set the maximum value for the color scale
    )
    plt.colorbar(label="Reflectivity")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Pump Delay (ps)")
    plt.title("Reflectivity Heatmap (clipped at {})".format(max_reflectivity))
    plt.savefig(output_file)
    plt.show()

    print(f"Heatmap plot saved to {output_file}")
    print(f"Heatmap data saved to {csv_output_file}")

    return heatmap_df

def calculate_reflectivity(E_Ref, E_Pump, time_axis):
    """
    Calculates reflectivity as a function of frequency and pump delay.
    """
    # Compute the frequency axis
    dt = time_axis[1] - time_axis[0]  # Time step
    freqs = np.fft.fftfreq(len(time_axis), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Keep only positive frequencies

    # Compute reflectivity
    E_Ref_Amp = np.abs(np.fft.fft(E_Ref, axis=0))[:len(freqs), :]
    E_Pump_Amp = np.abs(np.fft.fft(E_Pump, axis=0))[:len(freqs), :]
    E_Ref_Phase = -np.unwrap(np.angle(np.fft.fft(E_Ref, axis=0)))[:len(freqs), :]
    E_Pump_Phase = -np.unwrap(np.angle(np.fft.fft(E_Pump, axis=0)))[:len(freqs), :]
    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi
    R = R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))
    return np.abs(R),  freqs


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

def plot_reflectivity_heatmap(reflectivity, freqs, pump_delays, output_file,csv_output_file, max_freq=10):
    """
    Plots a 2D heat map of reflectivity with frequency on the x-axis.
    """

    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask] 
    reflectivity = reflectivity[freq_mask, :]
    # Save the heatmap data as a CSV file
    heatmap_df = pd.DataFrame(
        data=reflectivity.T,
        index=np.round(pump_delays, 2),
        columns=np.round(freqs, 2),
    )
    heatmap_df.index.name = "Pump Delay (ps)"
    heatmap_df.columns.name = "Frequency (THz)"
    heatmap_df.to_csv(csv_output_file)

    print(f"Heatmap plot saved to {output_file}")
    print(f"Heatmap data saved to {csv_output_file}")


    plt.figure(figsize=(12, 8))
    extent = [freqs[0], freqs[-1], pump_delays[0], pump_delays[-2]]
    plt.imshow(reflectivity.T, aspect="auto", extent=extent, origin="lower", cmap="viridis")
    plt.colorbar(label="Reflectivity")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Pump Delay (ps)")
    plt.title("Reflectivity Heatmap")
    plt.savefig(output_file)
    return heatmap_df

def find_large_reflectivity(heatmap_df, threshold=5):
    """
    Finds time delay and frequency pairs where the reflectivity exceeds a given threshold.

    Parameters:
    - heatmap_df: DataFrame of the heatmap, where rows are pump delays, columns are frequencies, and values are reflectivity.
    - threshold: Reflectivity threshold to exceed.

    Returns:
    - results: DataFrame with time delay, frequency, and reflectivity values exceeding the threshold.
    """
    results = []

    # Iterate through the DataFrame
    for pump_delay, row in heatmap_df.iterrows():
        for freq, reflectivity in row.items():
            if reflectivity > threshold:
                results.append({
                    "Pump Delay (ps)": pump_delay,
                    "Frequency (THz)": freq,
                    "Reflectivity": reflectivity,
                })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    Epump_file = "processed_data_roomtemp/E_pump.csv"
    Eref_file = "processed_data_roomtemp/E_ref.csv"
    output_folder = "reflection_analysis_roomtemp"
    os.makedirs(output_folder, exist_ok=True)

    E_Ref, E_Pump, time_axis = get_data(Epump_file, Eref_file)
    E_Ref_windowed, E_Pump_windowed, time_axis = process_data(E_Ref, E_Pump, time_axis)
    # Calculate differential reflectivity
    dreflectivity, freqs = calculate_differential_reflectivity( E_Ref_windowed, E_Pump_windowed, time_axis)
    # Calculate reflectivity
    R, freqs = calculate_reflectivity( E_Ref_windowed, E_Pump_windowed, time_axis)
    print(R)
    # Define pump delays
    initial_delay = -0.4  # Start pump delay in ps
    final_delay = 2.0    # End pump delay in ps
    step_size = 0.05     # Step size in ps
    additional_delays = "110"  # Additional delays, comma-separated

    # Parse additional delays and combine with regular delays
    additional_delays = [float(x) for x in additional_delays.split(",")] if additional_delays else []
    pump_delays = list(np.arange(initial_delay, final_delay + step_size, step_size)) + additional_delays
    pump_delays.sort()

    output_file = os.path.join(output_folder, "reflectivity_heatmap.png")
    csv_output_file = os.path.join(output_folder,"reflectivity_2d.csv")
    heatmap_df = plot_reflectivity_heatmap_clipped(np.abs(R), freqs, pump_delays, output_file, csv_output_file, max_freq=10)
    threshold = 10
    # Find points where reflectivity exceeds the threshold
    results = find_large_reflectivity(heatmap_df, threshold)
    fft_output_file = os.path.join(output_folder, "fft_heatmap_pump_delay.png")
    fft_csv_output_file = os.path.join(output_folder, "fft_heatmap_pump_delay.csv")

    fft_map_df = plot_fft_pump_time_axis(
        np.abs(R), pump_delays, freqs, fft_output_file, fft_csv_output_file, max_freq=10, max_fft=5
    )
