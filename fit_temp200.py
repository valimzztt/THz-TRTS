import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import scipy.constants as const
from src.refractive import *
from src.utils import *
from src.conductivity import *
from src.fitting import *
from src.reflectivity import *

# Constants
d = 1e-6 # Penetration depth (m) (700nm)
w = 2 * np.pi   # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
tauD = 3.5e-12  # Debye relaxation time (s)
magD = 80  # Debye magnitude
epsi = 8.6531 # Static permittivity for CdTe (Real part of refractive index)
eps_inf = epsi

def calculate_sigma_drude_lorentz(tau, wp, freq, w0):
    """
    Drude-Lorentz conductivity model:
    - Drude term: sigma_drude = wp^2 * tau * eps0 / (1 - i * freq * tau)
    - Lorentz term: sigma_lorentz = (wp_lorentz^2 * i * freq) / [(w0^2 - freq^2) + i * freq * gamma]

    Args:
        tau (float): Scattering time for Drude model (s).
        wp (float): Plasma frequency for Drude model (rad/s).
        freq (numpy array): Frequency array (Hz).
        w0 (float): Resonance frequency for Lorentz oscillator (rad/s).
        gamma (float): Damping constant for Lorentz oscillator (rad/s).
        wp_lorentz (float): Plasma frequency for Lorentz term (rad/s).

    Returns:
        sigma (numpy array): Total complex conductivity (S/m).
    """
    omega = 2 * np.pi * freq  # Angular frequency (rad/s)
    
    # Drude conductivity
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    
    # Lorentz conductivity
    sigma_lorentz = const.epsilon_0 * wp**2 / (
        (w0**2 - omega**2) - 1j  * omega * (1/tau)
    )
    # Total conductivity
    sigma = sigma_drude + sigma_lorentz
    
    return sigma






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



def calculate_sigma(r_tilde, n_tilde):
    """
    Calculate the complex photoconductivity from transient reflectivity.

    Parameters:
    - r_tilde: Complex reflectivity (numpy array).
    - n_tilde: Complex refractive index of the unpumped sample (numpy array).
    - Z0: Free-space impedance (Ohms).
    - d: Thickness of the photoconductive layer (m).

    Returns:
    - sigma: Complex photoconductivity (numpy array).
    """
    numerator = (1 - n_tilde**2) * (1 - r_tilde)
    denominator = Z0 * d * (r_tilde * (1 - n_tilde) + (1 + n_tilde))
    sigma = numerator / denominator
    return sigma




if __name__ == "__main__":
    lockin1_file = "processed_data_temp200/LockIn1.csv"
    lockin2_file = "processed_data_temp200/LockIn2.csv"
    output_folder = "reflection_analysis_temp200"
    pump_delay = "0.3"
    plotting = True

    E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column=pump_delay )
    #E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis = extend_to_power_of_2(E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis)
    R, E_Ref_FFT, E_Pump_FFT, freqs = analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis)
    initial_guess =  [1.00e-13,  6.8*1e13] # initial guess for tau and wp
    tau_fit, wp_fit = fit_transient_reflection(R, freqs, initial_guess)
    print(f"Pump delay: {pump_delay} ps")
    print(f"Fitted tau: {tau_fit:.2e} s")
    print(f"Fitted wp: {wp_fit:.2e} rad/s")

    
    # Drude-Lorentz model for the static reflection function 
    n_static_real, n_static_imag = unpumped_refractive_index("refractive_index_data_dec1.csv")
    wp_static = 1.79e+13 # rad/s
    gamma = 3.10e+12 
    wp_j = 5.81e+13 # Fitted Lorentz oscillator strength
    w0_j = 2.47e+13 
    gamma_j = 3.81e+12 
    n_static_real, n_static_imag = n_static_drude_lorentz(freqs, eps_inf, wp_static, gamma, wp_j, w0_j, gamma_j)
    n_static =  n_static_real+1j*n_static_imag
    omega_0 = w0_j
    # Calculate theoretical reflectivity
    sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs) # The photoexcited part can be described by a Drude Lorentz model
    eps_drude = eps_inf - wp_fit**2 / ((w*freqs)**2 + 1j * (w*freqs) / tau_fit)
    # Lorentz term
    eps_lorentz = wp_fit**2 / (w0**2 - (w*freqs)**2 - 1j * (w*freqs) / tau_fit)

    eps_fit = eps_inf + eps_lorentz + eps_drude
    n_fit = np.sqrt(eps_fit)

    R_calculated = ((1 - n_fit - Z0 * sigma_drude_fit * d) /
                        (1 + n_fit + Z0 * sigma_drude_fit * d)) * ((1 + n_fit) / (1 - n_fit))
    R_calculated = ((1 - n_static - Z0 * sigma_drude_fit * d) /
                        (1 + n_static + Z0 * sigma_drude_fit * d)) * ((1 + n_static) / (1 - n_static))
    R_experimental = calculate_reflection_coefficient(E_Pump_FFT-E_Ref_FFT, E_Ref_FFT)
    conductivity = calculate_sigma(R_experimental ,n_static)
    conductivity_lorentz = conductivity_with_resonance(freqs, omega_0, tau_fit, wp_fit)
    conductivity_total = conductivity_lorentz + sigma_drude_fit
    # Calculate photoinduced conductivity
    dr, r0 = photoinduced_conductivity(sigma_drude_fit,  Z0, d, n_static_real, n_static_imag)
    # Plot calculated and experimental reflectivity
    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e12, np.real(dr), label="Fitted Conductivity")
    plt.plot(freqs / 1e12, np.imag(dr), label="Fitted Conductivity")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Reflectivity")
    plt.title(f"Reflectivity Comparison (Pump Delay = {pump_delay} ps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Plot calculated and experimental reflectivity
    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e12, np.real(R ), label="Real(R_exp)")
    plt.plot(freqs / 1e12, np.imag(R), label="Imag(R_exp)") 
    plt.plot(freqs / 1e12, np.real(R_calculated), label="Real(R_fit)", linestyle="--")
    plt.plot(freqs / 1e12, np.imag(R_calculated), label="Imag(R_fit)", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
