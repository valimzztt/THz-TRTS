import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import scipy.constants as const

# Constants
d = 1.3e-6  # Penetration depth (m)
w = 2 * np.pi   # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
e = 1.602176565e-19  # Electron charge (C)
m = 9.10938291e-31  # Electron mass (kg)
eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
epsi = 8  # Static permittivity for CdTe
gam = 3.61e+13   # Damping constant (Hz)
w0 = 5.0 * w  # Resonance frequency (rad/s)
wpL = 1.0e14 # Plasma frequency (rad/s)
tau = 1.0e-13  # Relaxation time (s)
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
    omega = 2 * np.pi * freq
    return const.epsilon_0* tau * wp**2 / (1 - 1j *tau* omega)


def calculate_eps(freq, tau, wp):
    eps_debye = magD / (1 - 1j * freq * w * tauD)
    eps_lorentz = 1j * wpL**2 / (eps0 * (1j * ((w0**2 - (w*freq)**2)) + w*freq * gam))
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    eps_drude = sigma_drude / (1j * eps0 * freq * w)
    return epsi + eps_debye + eps_lorentz

def calculate_reflectivity(n, sigma_drude):
    r0 = (1 - n) / (1 + n)
    Re = -(1+r0)/r0 * ( Z0*d*sigma_drude ) / ( 1 + n + Z0*d*sigma_drude ) + 1
    return r0, Re


def calculate_transient_photoconductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, r0):
    """
    Calculate the optical conductivity using the Drude model.
    
    Parameters:
    - E_pump (array): Electric field (pump, reflected from sample).
    - E_ref (array): Electric field (reference, reflected from substrate).
    - r0 (complex): Complex reflectivity coefficient of the unpumped sample.

    
    Returns:
    - delta_sigma (array): Transient sheet photoconductivity.
    """
    # Fit reflectivity data only for frequencies from 0 to 10 THz
    E_pump = E_Pump_Amp * (np.cos(E_Pump_Phase) + 1j * np.sin(E_Pump_Phase))
    E_ref = E_Ref_Amp * (np.cos(E_Ref_Phase) + 1j * np.sin(E_Ref_Phase))
 
    # Calculate the field difference
    delta_E = E_pump - E_ref
    # Calculate transient sheet photoconductivity using the formula
    delta_sigma = (-1 + r0) / Z0 * (delta_E / E_ref)
    return delta_sigma



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
    """  next_power_of_2 = 1024
    pad_length = next_power_of_2 - len(time_axis)
    E_Ref = np.pad(E_Ref, (0, pad_length), 'constant')
    E_Pump = np.pad(E_Pump, (0, pad_length), 'constant') 
    time_axis = np.pad(time_axis, (0, pad_length), 'constant')"""

  
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

    freqs = freqs*1e12

    # Filter out frequencies from 0 to 10 THz
    mask = (freqs > 0) & (freqs <= 10e12)
    E_Ref_Amp = E_Ref_Amp[mask]
    E_Pump_Amp = E_Pump_Amp[mask]
    E_Ref_Phase = E_Ref_Phase[mask]
    E_Pump_Phase = E_Pump_Phase[mask]
    freqs = freqs[mask]
    R_Amp = R_Amp[mask]
    Phi = Phi[mask]
    R = R[mask]
    return R, E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs


def fitfun(params, R_measured, freq):
    tau, wp = params
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    eps = calculate_eps(freq, tau, wp)
    n = np.sqrt(eps)
    R_calculated = (1-n-Z0*d*sigma_drude) / (1+n+Z0*d*sigma_drude) / ( (1-n)/(1+n) )
    #R_calculated, Re = calculate_reflectivity(n, sigma_drude)
    error = np.abs(R_measured - R_calculated)**2 + (np.angle(R_measured) - np.angle(R_calculated))**2
    return error.flatten()
def unpumped_refractive_index(file_path):   
    """
    Assumes the input file is a CSV wfile which has the following columns: Frequency (Hz), Real Refractive Index, Imaginary Refractive Index

    """
    data = pd.read_csv(file_path)
    real_refractive_index = data['Real Refractive Index']
    imaginary_refractive_index = data['Imaginary Refractive Index']
    return real_refractive_index, imaginary_refractive_index

def photoinduced_conductivity(sigma_drude, freqs, Z0, d, unpumped_refractive_index_file):
    n_real, n_imag  = unpumped_refractive_index(unpumped_refractive_index_file)
    n = n_real + 1j * n_imag
    r0 = (n-1)/(n+1)
    dr = 1-((1-r0)/r0)* (Z0*d*sigma_drude) / (1+n+Z0*d*sigma_drude)
    plt.figure(figsize=(12, 8))
    plt.plot(freqs / 1e12, dr, label="Real(dr)")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Reflectivity")
    plt.legend()
    plt.grid(True)
    #plt.show()
    return dr, r0
    
def fit_data(R, freqs, init):
    # Fit reflectivity data
    bounds = ([1e-13, 1e12], [1e-11, 1e14])
    result = least_squares(fitfun, init, bounds=bounds, args=(R, freqs))
    tau_fit, wp_fit = result.x
    sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs)
    eps_fit = calculate_eps(freqs, tau_fit, wp_fit) + sigma_drude_fit / (1j * eps0 * freqs)
    n_fit = np.sqrt(eps_fit)
    return n_fit, sigma_drude_fit, tau_fit, wp_fit


if __name__ == "__main__":
    lockin1_file = "processed_data_roomtemp/LockIn1.csv"
    lockin2_file = "processed_data_roomtemp/LockIn2.csv"
    output_folder = "reflection_analysis_roomtemp"
    pump_delay = "0.3"
    plotting = True
    E_Ref_windowed, E_Pump_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column=pump_delay )
    R, E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs= analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis)
    initial_guess =  [(1/9.85e+12),  6.8*1e13] # initial guess for tau and wp
    n_fit, sigma_drude_fit, tau_fit, wp_fit = fit_data(R, freqs,initial_guess)
    print(f"Fitted tau: {tau_fit:.2e}, Fitted wp: {wp_fit:.2e}")
    dr, r0 = photoinduced_conductivity(sigma_drude_fit, freqs, Z0, d, unpumped_refractive_index_file="refractive_index_data_dec1.csv")
    # Calculate transient photoconductivity
    delta_sigma = calculate_transient_photoconductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase,  r0)
    if plotting==True:
        # Plot Sigma Drude
        plt.figure(figsize=(12, 8))
        plt.title("Drude Conductivity, pump delay= " + pump_delay + " ps")
        plt.plot(freqs / 1e12, np.imag(sigma_drude_fit), label="Imag(sigma_drude)")
        plt.plot(freqs / 1e12, np.real(sigma_drude_fit), label="Real(sigma_drude)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Drude conductivity (S/m)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("Transient Photoconductivity, pump delay= " + pump_delay + " ps")
        plt.plot(freqs / 1e12, np.imag(delta_sigma), label="Imag(delta_sigma)")
        plt.plot(freqs / 1e12, np.real(delta_sigma), label="Real(delta_sigma)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transient Photoconductivity (S/m)")
        plt.legend()
        plt.grid(True)
        plt.show()