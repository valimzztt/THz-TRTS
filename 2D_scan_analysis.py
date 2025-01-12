import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Slider

# Constants
d = 1.3e-6  # Penetration depth (m)
w = 2 * np.pi * 1e12  # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
e = 1.602176565e-19  # Electron charge (C)
m = 9.10938291e-31  # Electron mass (kg)
eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
epsi = 10.2  # Static permittivity for CdTe
gam = 1 * 1e11  # Damping constant (Hz)
w0 = 5.0 * w  # Resonance frequency (rad/s)
wpL = 1.65e8  # Plasma frequency (rad/s)
tauD = 3.5e-12  # Debye relaxation time (s)
magD = 80  # Debye magnitude


def calculate_sigma_drude(tau, wp, freq):
    """
    Drude conductivity formulas:
    DC conductivity: sigma_dc = n * e**2 * tau / m
    Plasma frequency: wp = sqrt(n * e**2 / (m * eps0))
    n * e**2 = wp**2 * m * eps0
    Drude conductivity: sigma_drude = sigma_dc  / (1 - 1j * freq * w * tau)
    Drude conductivity: sigma_drude =  wp**2 * tau  * eps0  / (1 - 1j * freq * w * tau)
    """
    return eps0 * tau * wp**2 / (1 - 1j * freq * tau * w)

def calculate_eps(freq, tau, wp):
    eps_debye = magD / (1 - 1j * freq * w * tauD)
    eps_lorentz = 1j * wpL**2 / (eps0 * (1j * ((w0**2 - freq**2)) + freq * gam))
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    eps_drude = sigma_drude / (1j * eps0 * freq * w)
    return epsi + eps_debye + eps_lorentz + eps_drude

def calculate_reflectivity(n, sigma_drude):
    r0 = (1 - n) / (1 + n)
    Re = -(1+r0)/r0 * ( Z0*d*sigma_drude ) / ( 1 + n + Z0*d*sigma_drude ) + 1
    return r0, Re


def fitfun(params, R_measured, freq):
    tau, wp = params
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    eps = calculate_eps(freq, tau, wp)
    print(eps)
    n = np.sqrt(eps)
    R_calculated, Re = calculate_reflectivity(n, sigma_drude)
    error = np.abs(R_measured - R_calculated)**2 + (np.angle(R_measured) - np.angle(R_calculated))**2
    return error.flatten()



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

    #plt.show() 
    plt.close()   
    return E_Ref_windowed, E_Pump_windowed, time_axis




def analyze_reflection(E_Ref_windowed,E_Pump_windowed,time_axis ):
    """
    Analyze the reflection data using the windowed E_Ref and E_Pump signals.

    Parameters:
    - E_Ref_windowed (np.ndarray): Windowed E_Ref signal.
    - E_Pump_windowed (np.ndarray): Windowed E_Pump signal.
    """
    E_Ref = E_Ref_windowed
    E_Pump = E_Pump_windowed
    # Add zero baseline to signals
    next_power_of_2 = 1024
    pad_length = next_power_of_2 - len(time_axis)
    E_Ref = np.pad(E_Ref, (0, pad_length), 'constant')
    E_Pump = np.pad(E_Pump, (0, pad_length), 'constant')
    time_axis = np.pad(time_axis, (0, pad_length), 'constant')
    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]  # Positive frequencies only

    E_Ref_Amp = np.abs(np.fft.fft(E_Ref))[:len(freqs)]
    E_Pump_Amp = np.abs(np.fft.fft(E_Pump))[:len(freqs)]
    E_Ref_Phase = -np.unwrap(np.angle(np.fft.fft(E_Ref)))[:len(freqs)]
    E_Pump_Phase = -np.unwrap(np.angle(np.fft.fft(E_Pump)))[:len(freqs)]
    
    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi

    R = R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, R_Amp, label="R_Amp")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Reflectivity Amplitude")
    plt.legend()
    plt.xlim(0,20)
    plt.grid(True)
    plt.title("Reflectivity vs Frequency")
    plt.subplot(2, 1, 2)
    plt.plot(freqs, Phi, label="R_Amp")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase")
    plt.legend()
    plt.xlim(0,20)
    plt.grid(True)
    #plt.show()
    plt.close()
    return R, E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs
    
def fit_data(R, freqs):
    # Fit reflectivity data only for frequencies from 0 to 10 THz
    freqs = freqs*1e12
    mask = (freqs > 0) & (freqs <= 10e12)
    R = R[mask]
    freqs = freqs[mask]

    # Fit reflectivity data
    init = [3.5e-15, 5e13]  # [tau, wp]
    bounds = ([1e-15, 1e13], [1e-12, 1e14])

    result = least_squares(fitfun, init, bounds=bounds, args=(R, freqs))
    tau_fit, wp_fit = result.x
    print(tau_fit, wp_fit)

    sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs)
    eps_fit = calculate_eps(freqs, tau_fit, wp_fit) + sigma_drude_fit / (1j * eps0 * freqs)
    n_fit = np.sqrt(eps_fit)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, sigma_drude_fit, label="Measured Reflectivity")
    #plt.plot(freqs / 1e12, np.abs(eps_fit), label="eps_fit", linestyle="--")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Drude conductivity")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs / 1e12, np.real(n_fit), label="Real(n)")
    plt.plot(freqs / 1e12, np.imag(n_fit), label="Imaginary(n)")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive Index")
    plt.legend()
    plt.grid(True)
    plt.title("Refractive Index vs Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "reflectivity_analysis.png"))
    plt.show()

def calculate_optical_conductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs, eps_inf=10.2):
    """
    Calculate the optical conductivity using the Drude model.
    
    Parameters:
    - E_pump (array): Electric field (pump, reflected from sample).
    - E_ref (array): Electric field (reference, reflected from substrate).
    - freq (array): Frequency axis (Hz).
    - eps_inf (float): High-frequency permittivity.
    
    Returns:
    - sigma (array): Complex optical conductivity (S/m).
    """
    # Fit reflectivity data only for frequencies from 0 to 10 THz
    freqs = freqs*1e12
    mask = (freqs > 0) & (freqs <= 10e12)
    freqs = freqs[mask]
    E_pump = E_Pump_Amp[mask] * (np.cos(E_Pump_Phase[mask]) + 1j * np.sin(E_Pump_Phase[mask]))
    E_ref = E_Ref_Amp[mask] * (np.cos(E_Ref_Phase[mask]) + 1j * np.sin(E_Ref_Phase[mask]))
    # Calculate transfer function
    H = E_pump / E_ref

    # Compute complex refractive index
    n_tilde = (1 - H) / (1 + H)

    # Compute complex permittivity
    eps_tilde = n_tilde**2

    # Compute complex optical conductivity
    omega = 2 * np.pi * freqs  # Angular frequency
    sigma = 1j * eps0 * omega * (eps_inf - eps_tilde)
    plt.figure(figsize=(12, 8))
    plt.plot(freqs, np.real(sigma), label="Measured Condictivity")
    plt.plot(freqs, np.imag(sigma), label="Measured Condictivity")

    #plt.plot(freqs / 1e12, np.abs(eps_fit), label="eps_fit", linestyle="--")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Drude conductivity")
    plt.legend()
    plt.grid(True)
    plt.show()
    return sigma, n_tilde, eps_tilde


if __name__ == "__main__":
    lockin1_file = "processed_data/LockIn1.csv"
    lockin2_file = "processed_data/LockIn2.csv"
    output_folder = "reflection_analysis"
    E_Ref_windowed, E_Pump_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column="0.3")
    R, E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs = analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis)
    fit_data(R, freqs)
    calculate_optical_conductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs, eps_inf=10.2)

