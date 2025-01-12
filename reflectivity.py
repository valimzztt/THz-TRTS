import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def process_data(lockin1_file, lockin2_file, time_delay_column):
    """
    A function that processes the data from the Lock-In Amplifier CSV files and save the E_Ref and E_Pump signals.
    Functionalities include baseline subtraction, windowing, and Fourier transform.
    Parameters:
    - lockin1_file (str): Path to the LockIn1 CSV file.
    - lockin2_file (str): Path to the LockIn2 CSV file.
    - time_delay_column (str): Column name for the time delay data.
    """
    # Load data from CSV files
    lockin1_data = pd.read_csv(lockin1_file)
    lockin2_data = pd.read_csv(lockin2_file)
    # Extract time axis and specific pump delay
    global time_axis
    time_axis = lockin1_data.iloc[:, 0].values  # First column is time
    lockin1_signals = lockin1_data[time_delay_column].values
    lockin2_signals = lockin2_data[time_delay_column].values
    # Subtract baselines
    lockin1_signals -= np.mean(lockin1_signals)
    lockin2_signals -= np.mean(lockin2_signals)
  
    # Compute reflection-related quantities
    E_Ref = lockin1_signals - lockin2_signals
    E_Pump = lockin1_signals + lockin2_signals
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
    # Generate Hann windows centered at the specified positions
    ref_hann_window = create_centered_hann(len(E_Ref),hann_window_center, hann_window_width)
    pump_hann_window = create_centered_hann(len(E_Pump), hann_window_center, hann_window_width)
    E_Ref_windowed = E_Ref * ref_hann_window
    E_Pump_windowed = E_Pump * pump_hann_window

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    axs[0, 0].plot(time_axis, E_Ref, label="E_ref")
    axs[0, 0].plot(time_axis, E_Ref_windowed, label="E_ref Hann Window")
    axs[0, 0].plot(time_axis, ref_hann_window, label="Hann Window")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_title("E_ref vs Time")

    axs[0, 1].plot(time_axis, E_Pump, label="E_pump")
    axs[0, 1].plot(time_axis, pump_hann_window,label="Hann Window")
    axs[0, 1].plot(time_axis, E_Pump_windowed,label="E_pump Hann Window")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_title("E_pump vs Time")

    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]  # Positive frequencies only
    # Compute Fourier transforms of windowed and unwindowed signals
    E_Ref_FFT = np.fft.fft(E_Ref)[:len(freqs)]
    E_Pump_FFT = np.fft.fft(E_Pump)[:len(freqs)]
    E_Ref_windowed_FFT = np.fft.fft(E_Ref_windowed)[:len(freqs)]
    E_Pump_windowed_FFT = np.fft.fft(E_Pump_windowed)[:len(freqs)]

    axs[1,0].plot(freqs, np.abs(E_Ref_FFT), label="E_ref FFT")
    axs[1,0].plot(freqs, np.abs(E_Ref_windowed_FFT), label="E_ref Windowed FFT")
    axs[1, 0].set_xlabel("Frequency (THz)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].legend()
    axs[1, 0].set_xlim(0,20)
    axs[1, 0].grid(True)
    axs[1, 0].set_title("E_ref FFT vs Frequency")

    axs[1, 1].plot(freqs, np.abs(E_Pump_FFT), label="E_pump FFT")
    axs[1, 1].plot(freqs, np.abs(E_Pump_windowed_FFT), label="E_pump Windowed FFT")
    axs[1, 1].set_xlabel("Frequency (THz)")
    axs[1, 1].set_ylabel("Amplitude")
    axs[1, 1].set_xlim(0,20)
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_title("E_pump FFT vs Frequency")

    plt.close()   
    return E_Ref_windowed, E_Pump_windowed, time_axis



def reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase):
    """
    Calculate the reflectivity from the reference and pump amplitudes and phases.
    Args:
        E_Ref_Amp (array): Amplitude of the reference signal.
        E_Pump_Amp (array): Amplitude of the pump signal.
        E_Ref_Phase (array): Phase of the reference signal.
        E_Pump_Phase (array): Phase of the pump signal.
    Returns:
        R (array): Reflectivity.
    """
    R = (E_Pump_Amp / E_Ref_Amp) * np.exp(1j * (E_Pump_Phase - E_Ref_Phase))
    return R


import csv


def differential_reflectivity(E_Ref_windowed, E_Pump_windowed, file_path):
    """
    Compute and plot the differential reflectivity amplitude |Δr(ω, t)| / |r0| at a target frequency.
    Args:
        freq (array): Frequency axis from FFT.
        
    """
    # Open the file and read the header
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the first row (header)
        pump_delays = np.array(header[1:-1], dtype=float)  # Convert pump delays to a numpy array of floats
  

    # Compute FFT of the windowed signals
    E_Ref_FFT = np.fft.fft(E_Ref_windowed)
    E_Pump_FFT = np.fft.fft(E_Pump_windowed)
    
    # Frequency axis
    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]  # Positive frequencies only
    
    # Initialize differential reflectivity array
    diff_reflectivity = np.zeros((len(pump_delays), len(freqs)))
    
    for i, delay in enumerate(pump_delays):
        # Compute the amplitude and phase of the FFTs
        E_Ref_Amp = np.abs(E_Ref_FFT[:len(freqs)])
        E_Pump_Amp = np.abs(E_Pump_FFT[:len(freqs)])
        E_Ref_Phase = np.angle(E_Ref_FFT[:len(freqs)])
        E_Pump_Phase = np.angle(E_Pump_FFT[:len(freqs)])
        
        # Compute reflectivity
        R = reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase)
        
        # Compute differential reflectivity
        R0 = reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase)  # Reflectivity at 0th pump delay
        diff_reflectivity[i, :] = (np.abs(R - R0) / np.abs(R0))
    
    # Plot the differential reflectivity as a 2D map
    plt.figure(figsize=(10, 6))
    plt.imshow(diff_reflectivity, aspect='auto', extent=[freqs[0], freqs[-1], pump_delays[-1], pump_delays[0]], cmap='viridis')
    plt.colorbar(label='Differential Reflectivity')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Pump Delay (s)')
    plt.title('Differential Reflectivity Map')
    plt.show()


if __name__ == "__main__":
    lockin1_file = "processed_data/LockIn1.csv"
    lockin2_file = "processed_data/LockIn2.csv"
    output_folder = "reflection_analysis"
    E_Ref_windowed, E_Pump_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column="0.3")
    differential_reflectivity(E_Ref_windowed, E_Pump_windowed, file_path = "processed_data/LockIn1.csv")

    #calculate_optical_conductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs, eps_inf=10.2)
