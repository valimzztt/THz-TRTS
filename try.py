import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def load_data(folder, time_step, start_delay):
    """
    Load data files from a folder. Each file contains time, V1, and V2 columns.
    Args:
        folder (str): Path to the folder containing data files.
        time_step (float): Time step between pump delays.
        start_delay (float): Starting pump delay time.
    Returns:
        times (array): Time axis.
        pump_delays (list): List of pump delays.
        V1_list, V2_list (list): Lists of V1 and V2 data for each pump delay.
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith('.txt') or f.endswith('.d24')])
    pump_delays = [start_delay + i * time_step for i in range(len(files))]
    V1_list, V2_list = [], []

    for file in files:
        data = np.loadtxt(os.path.join(folder, file))
        times = data[:, 0]  # Time column
        V1_list.append(data[:, 1])  # V1 column
        V2_list.append(data[:, 2])  # V2 column

    return times, pump_delays, V1_list, V2_list

def baseline_correction(V_list):
    """
    Perform baseline correction by subtracting the mean from each signal.
    """
    return [V - np.mean(V) for V in V_list]

def compute_signals(V1_list, V2_list):
    """
    Compute E_Ref, E_Pump, and dE from V1 and V2.
    """
    E_Ref = [V1 - V2 for V1, V2 in zip(V1_list, V2_list)]
    E_Pump = [V1 + V2 for V1, V2 in zip(V1_list, V2_list)]
    dE = [2 * V2 for V2 in V2_list]
    return E_Ref, E_Pump, dE

def fft_analysis(E_signal, time):
    """
    Perform FFT analysis to compute amplitude and phase.
    Args:
        E_signal (array): Time-domain signal.
        time (array): Time axis.
    Returns:
        freq (array): Frequency axis.
        amplitude (array): FFT amplitude.
        phase (array): FFT phase.
    """
    N = len(time)
    time_step = time[1] - time[0]
    freq = fftfreq(N, time_step)
    fft_signal = fft(E_signal)
    amplitude = 2 * np.abs(fft_signal) / N
    phase = -np.unwrap(np.angle(fft_signal))
    return freq[:N // 2], amplitude[:N // 2], phase[:N // 2]

def reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase):
    """
    Compute the reflectivity based on amplitude and phase of FFT signals.
    """
    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    return R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))
def differential_reflectivity(freq, pump_delays, E_Ref_list, E_Pump_list, time, ref_index=0, target_freq=5.0, max_freq=10):
    """
    Compute and plot the differential reflectivity amplitude |Δr(ω, t)| / |r0| at a target frequency.
    Args:
        freq (array): Frequency axis from FFT.
        pump_delays (list): List of pump delays.
        E_Ref_list, E_Pump_list (list): FFT input signals.
        time (array): Time axis.
        ref_index (int): Index of the reference reflectivity (e.g., 0th pump delay).
        target_freq (float): Frequency of interest in THz.
        max_freq (float): Maximum frequency for visualization.
    """
    reflectivity_list = []
    for E_Ref, E_Pump in zip(E_Ref_list, E_Pump_list):
        freq, E_Ref_Amp, E_Ref_Phase = fft_analysis(E_Ref, time)
        _, E_Pump_Amp, E_Pump_Phase = fft_analysis(E_Pump, time)
        R = reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase)
        reflectivity_list.append(R)

    reflectivity_list = np.array(reflectivity_list)
    r0 = reflectivity_list[ref_index]  # Reference reflectivity at ref_index

    # Compute differential reflectivity amplitude
    delta_r = np.abs(reflectivity_list - r0) / np.abs(r0)

    # Extract target frequency
    freq_THz = freq
    if not (freq_THz.min() <= target_freq <= freq_THz.max()):
        raise ValueError(f"Target frequency {target_freq} THz is outside the range [{freq_THz.min():.2f}, {freq_THz.max():.2f}] THz")

    target_index = np.argmin(np.abs(freq_THz - target_freq))  # Find closest frequency index

    delta_r_target = delta_r[:, target_index]

    # Plot the differential reflectivity vs pump delay
    plt.figure(figsize=(8, 6))
    plt.plot(pump_delays, delta_r_target, 'o-', label=f'{target_freq:.1f} THz')
    plt.xlabel('Pump Delay (ps)')
    plt.ylabel(r'$|\Delta r / r_0|$')
    plt.title(f'Differential Reflectivity at {target_freq:.1f} THz')
    plt.legend()
    plt.grid()
    plt.show()


def plot_2D_map(freq, pump_delays, E_Ref_list, E_Pump_list, time, max_freq=10):
    """
    Plot a 2D map of reflectivity with x-axis as frequency (up to max_freq in THz) and y-axis as pump delays.
    Args:
        freq (array): Frequency axis from FFT.
        pump_delays (list): List of pump delays.
        E_Ref_list, E_Pump_list (list): FFT input signals.
        time (array): Time axis.
        max_freq (float): Maximum frequency to plot in THz.
    """
    reflectivity_list = []
    for E_Ref, E_Pump in zip(E_Ref_list, E_Pump_list):
        freq, E_Ref_Amp, E_Ref_Phase = fft_analysis(E_Ref, time)
        _, E_Pump_Amp, E_Pump_Phase = fft_analysis(E_Pump, time)
        R = reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase)
        reflectivity_list.append(np.abs(R))

    reflectivity_list = np.array(reflectivity_list)
    freq_THz = freq  # Convert to THz
    freq_mask = freq_THz <= max_freq

    # Plot the 2D map
    plt.figure(figsize=(12, 8))
    plt.imshow(reflectivity_list[:, freq_mask], aspect='auto',
               extent=[freq_THz[freq_mask][0], freq_THz[freq_mask][-1], pump_delays[-1], pump_delays[0]],
               cmap='viridis')
    plt.colorbar(label='Reflectivity')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Pump Delay (ps)')
    plt.title('2D Map of Reflectivity vs Pump Delay and Frequency')
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def load_data(folder, time_step, start_delay):
    """
    Load data files from a folder. Each file contains time, V1, and V2 columns.
    Args:
        folder (str): Path to the folder containing data files.
        time_step (float): Time step between pump delays.
        start_delay (float): Starting pump delay time.
    Returns:
        times (array): Time axis.
        pump_delays (list): List of pump delays.
        V1_list, V2_list (list): Lists of V1 and V2 data for each pump delay.
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith('.txt') or f.endswith('.d24')])
    pump_delays = [start_delay + i * time_step for i in range(len(files))]
    V1_list, V2_list = [], []

    for file in files:
        data = np.loadtxt(os.path.join(folder, file))
        times = data[:, 0]  # Time column
        V1_list.append(data[:, 1])  # V1 column
        V2_list.append(data[:, 2])  # V2 column

    return times, pump_delays, V1_list, V2_list

def baseline_correction(V_list):
    """
    Perform baseline correction by subtracting the mean from each signal.
    """
    return [V - np.mean(V) for V in V_list]

def compute_signals(V1_list, V2_list):
    """
    Compute E_Ref, E_Pump, and dE from V1 and V2.
    """
    E_Ref = [V1 - V2 for V1, V2 in zip(V1_list, V2_list)]
    E_Pump = [V1 + V2 for V1, V2 in zip(V1_list, V2_list)]
    dE = [2 * V2 for V2 in V2_list]
    return E_Ref, E_Pump, dE

def fft_analysis(E_signal, time):
    """
    Perform FFT analysis to compute amplitude and phase.
    Args:
        E_signal (array): Time-domain signal.
        time (array): Time axis.
    Returns:
        freq (array): Frequency axis.
        amplitude (array): FFT amplitude.
        phase (array): FFT phase.
    """
    N = len(time)
    time_step = time[1] - time[0]
    freq = fftfreq(N, time_step)
    fft_signal = fft(E_signal)
    amplitude = 2 * np.abs(fft_signal) / N
    phase = -np.unwrap(np.angle(fft_signal))
    return freq[:N // 2], amplitude[:N // 2], phase[:N // 2]

def reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase):
    """
    Compute the reflectivity based on amplitude and phase of FFT signals.
    """
    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    return R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))

def differential_reflectivity_map(freq, pump_delays, E_Ref_list, E_Pump_list, time, ref_index=0, max_freq=10):
    """
    Compute and plot a 2D map of differential reflectivity amplitude |Δr(ω, t)| / |r0|.
    Args:
        freq (array): Frequency axis from FFT.
        pump_delays (list): List of pump delays.
        E_Ref_list, E_Pump_list (list): FFT input signals.
        time (array): Time axis.
        ref_index (int): Index of the reference reflectivity (e.g., 0th pump delay).
        max_freq (float): Maximum frequency for visualization.
    """
    reflectivity_list = []
    for E_Ref, E_Pump in zip(E_Ref_list, E_Pump_list):
        freq, E_Ref_Amp, E_Ref_Phase = fft_analysis(E_Ref, time)
        _, E_Pump_Amp, E_Pump_Phase = fft_analysis(E_Pump, time)
        R = reflectivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase)
        reflectivity_list.append(R)

    reflectivity_list = np.array(reflectivity_list)
    r0 = reflectivity_list[ref_index]  # Reference reflectivity at ref_index

    # Compute differential reflectivity amplitude
    delta_r = np.abs(reflectivity_list - r0) / np.abs(r0)

    # Limit frequency range to max_freq
    freq_THz = freq  # Convert to THz
    freq_mask = freq_THz <= max_freq

    # Plot the 2D map of differential reflectivity
    plt.figure(figsize=(12, 8))
    plt.imshow(delta_r[:, freq_mask], aspect='auto',
               extent=[freq_THz[freq_mask][0], freq_THz[freq_mask][-1], pump_delays[-1], pump_delays[0]],
               cmap='viridis')
    plt.colorbar(label=r'$|\Delta r / r_0|$')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Pump Delay (ps)')
    plt.title('2D Map of Differential Reflectivity Amplitude')
    plt.show()

def main(folder, time_step, start_delay):
    # Load and process data
    times, pump_delays, V1_list, V2_list = load_data(folder, time_step, start_delay)
    print(pump_delays)
    V1_list = baseline_correction(V1_list)
    V2_list = baseline_correction(V2_list)
    E_Ref, E_Pump, _ = compute_signals(V1_list, V2_list)

    # FFT analysis (single example to get frequency)
    freq, _, _ = fft_analysis(E_Ref[0], times)

    # Plot the 2D reflectivity map
    plot_2D_map(freq, pump_delays, E_Ref, E_Pump, times)

    # Plot the differential reflectivity map
    differential_reflectivity_map(freq, pump_delays, E_Ref, E_Pump, times, ref_index=0, max_freq=10)

if __name__ == "__main__":
    folder = "Dec12_2D_Temp200_AveragedData"
    time_step = 0.05 #float(input("Enter the time step between pump delays (in ps): "))
    start_delay = -0.3 #float(input("Enter the starting pump delay (in ps): "))
    main(folder, time_step, start_delay)
