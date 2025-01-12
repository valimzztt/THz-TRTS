import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Constants
d = 1.3e-6  # Penetration depth (m)
w = 2 * np.pi * 1e12  # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
e = 1.602176565e-19  # Electron charge (C)
m = 9.10938291e-31  # Electron mass (kg)
eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
epsi = 10.2  # Static permittivity for CdTe
gam = 1 * 1e11  # Damping constant (Hz)
w0 = 4.5 * w  # Resonance frequency (rad/s)
wpL = 1.65e8  # Plasma frequency (rad/s)
tauD = 3.5e-12  # Debye relaxation time (s)
magD = 80  # Debye magnitude

# Functions
def calculate_sigma_drude(tau, wp, freq):
    return eps0 * tau * wp**2 / (1 - 1j * freq * tau)

def calculate_eps(freq):
    eps_debye = magD / (1 - 1j * freq * tauD)
    eps_lorentz = 1j * wpL**2 / (eps0 * (1j * ((w0**2 - freq**2)) + freq * gam))
    return epsi + eps_debye + eps_lorentz

def calculate_reflectivity(n, sigma_drude):
    r0 = (1 - n) / (1 + n)
    return -(1 + r0) / r0 * (Z0 * d * sigma_drude) / (1 + n + Z0 * d * sigma_drude) + 1

def fitfun(params, R_measured, freq):
    tau, wp = params
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    eps = calculate_eps(freq) + sigma_drude / (1j * eps0 * freq)
    n = np.sqrt(eps)
    R_calculated = calculate_reflectivity(n, sigma_drude)
    error = np.abs(R_measured - R_calculated)**2 + (np.angle(R_measured) - np.angle(R_calculated))**2
    return error.flatten()

def process_pump_delays(data_folder, output_folder, pump_delays):
    """
    Analyze reflection data for each pump delay.

    Parameters:
    - data_folder (str): Path to the folder containing .d24 files for each pump delay.
    - output_folder (str): Path to the folder where results will be saved.
    - pump_delays (list): List of pump delay times corresponding to each file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Find all .d24 files in the data folder
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".d24")])
    if len(files) != len(pump_delays):
        print(len(files))
        print(len(pump_delays))
        raise ValueError("Mismatch between number of pump delay files and pump delay times.")

    results = []

    for i, (file, delay) in enumerate(zip(files, pump_delays)):
        file_path = os.path.join(data_folder, file)
        print(f"Processing {file} ({i+1}/{len(files)}), Pump Delay: {delay} ps")

        # Read data
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Time", "LockIn1", "LockIn2"])

        # Extract signals and time axis
        time_axis = data["Time"].values
        lockin1_signal = data["LockIn1"].values
        lockin2_signal = data["LockIn2"].values

        # Zero-pad signals to 1024 points
        zero_pad_length = 1024
        time_axis_padded = np.linspace(time_axis[0], time_axis[-1], zero_pad_length)
        lockin1_signal_padded = np.pad(lockin1_signal, (0, zero_pad_length - len(lockin1_signal)), mode="constant")
        lockin2_signal_padded = np.pad(lockin2_signal, (0, zero_pad_length - len(lockin2_signal)), mode="constant")

        # Compute reflection-related quantities
        E_Ref = lockin1_signal_padded - lockin2_signal_padded
        E_Pump = lockin1_signal_padded + lockin2_signal_padded

        time_sample = time_axis_padded[1] - time_axis_padded[0]
        freqs = np.fft.fftfreq(len(time_axis_padded), d=time_sample)
        freqs = freqs[:len(freqs) // 2]  # Positive frequencies only

        E_Ref_Amp = np.abs(np.fft.fft(E_Ref))[:len(freqs)]
        E_Pump_Amp = np.abs(np.fft.fft(E_Pump))[:len(freqs)]
        E_Ref_Phase = -np.unwrap(np.angle(np.fft.fft(E_Ref)))[:len(freqs)]
        E_Pump_Phase = -np.unwrap(np.angle(np.fft.fft(E_Pump)))[:len(freqs)]

        R_Amp = E_Pump_Amp / E_Ref_Amp
        Phi = E_Pump_Phase - E_Ref_Phase
        Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi

        R = R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))

        # Fit reflectivity data
        init = [70e-15, 8.5e13]  # [tau, wp]
        bounds = ([0, 1e13], [1e-12, 1e14])

        result = least_squares(fitfun, init, bounds=bounds, args=(R, freqs))
        tau_fit, wp_fit = result.x

        sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs)
        eps_fit = calculate_eps(freqs) + sigma_drude_fit / (1j * eps0 * freqs)
        n_fit = np.sqrt(eps_fit)
        R_fit = calculate_reflectivity(n_fit, sigma_drude_fit)

        # Save results for each pump delay
        pump_output_file = os.path.join(output_folder, f"results_pump_delay_{delay:.2f}.csv")
        pd.DataFrame({
            "Frequency (THz)": freqs / 1e12,
            "Measured Reflectivity (abs)": np.abs(R),
            "Fitted Reflectivity (abs)": np.abs(R_fit),
            "Real(n)": np.real(n_fit),
            "Imaginary(n)": np.imag(n_fit)
        }).to_csv(pump_output_file, index=False)

        results.append((freqs, R, R_fit, n_fit))

    print(f"Analysis completed. Results saved in {output_folder}")

def animate_signals(data_folder, pump_delays):
    """
    Create an animation of E1 and E2 signals for each pump delay.

    Parameters:
    - data_folder (str): Path to the folder containing .d24 files for each pump delay.
    - pump_delays (list): List of pump delay times corresponding to each file.
    """
    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".d24")])
    if len(files) != len(pump_delays):
        raise ValueError("Mismatch between number of pump delay files and pump delay times.")

    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label="E1 (LockIn1)", color="blue")
    line2, = ax.plot([], [], label="E2 (LockIn2)", color="orange")
    ax.set_xlim(0, 10)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Signal")
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(frame):
        file_path = os.path.join(data_folder, files[frame])
        
        # Read the file, skipping the first line (header information)
        data = pd.read_csv(
            file_path,
            delim_whitespace=True,
            skiprows=1,  # Skip the first line of metadata
            header=None,  # No header in the actual data
            names=["Time", "LockIn1", "LockIn2"]  # Explicitly define column names
        )

        # Strip any extra whitespace from column names (if needed)
        data.columns = data.columns.str.strip()

        # Extract columns for plotting
        time_axis = data["Time"].values
        lockin1_signal = data["LockIn1"].values
        lockin2_signal = data["LockIn2"].values

        # Update the line plots with new data
        line1.set_data(time_axis, lockin1_signal)
        line2.set_data(time_axis, lockin2_signal)
        
        # Update the plot title with the pump delay
        ax.set_title(f"Pump Delay: {pump_delays[frame]:.2f} ps")
        ax.relim()
        ax.autoscale_view() 
        
        return line1, line2


    ani = FuncAnimation(fig, update, frames=len(files), init_func=init, blit=True)
    plt.show()

if __name__ == "__main__":
    data_folder = "Dec12_2D_Temp200_AveragedDataNew"
    output_folder = "reflection_analysis_new"

    # Define pump delays
    initial_delay = -0.3  # Start pump delay in ps
    final_delay = 2.0    # End pump delay in ps
    step_size = 0.05     # Step size in ps
    additional_delays = "110"  # Additional delays, comma-separated

    # Parse additional delays and combine with regular delays
    additional_delays = [float(x) for x in additional_delays.split(",")] if additional_delays else []
    pump_delays = list(np.arange(initial_delay, final_delay + step_size, step_size)) + additional_delays
    pump_delays.sort()

    # Ensure directories exist
    os.makedirs(output_folder, exist_ok=True)

    # Run animation for E1 and E2 signals
    print("Creating animation of E1 and E2 signals...")
    animate_signals(data_folder, pump_delays)

    # Run analysis for each pump delay
    print("Processing pump delays and analyzing reflectivity...")
    process_pump_delays(data_folder, output_folder, pump_delays)

    print("All tasks completed successfully!")
