import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import scipy.constants as const
from src.refractive import *
from src.utils import *

# Constants
d = 7e-7 # Penetration depth (m) (700nm)
w = 2 * np.pi   # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
e = 1.602176565e-19  # Electron charge (C)
m = 9.10938291e-31  # Electron mass (kg)
tauD = 3.5e-12  # Debye relaxation time (s)
magD = 80  # Debye magnitude
eps_inf =  10.4
res_freq_THz = 5
w0 = res_freq_THz*w*1e12  # Resonance frequency (rad/s)

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
    return const.epsilon_0*tau * wp**2 / (1 - 1j *tau* omega)

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



def calculate_eps(freq, tau, wp):
    omega = 2 * np.pi * freq  # Angular frequency (rad/s)
    
    # Drude term
    eps_drude = 1 - wp**2 / (omega**2 + 1j * omega / tau)
    
    # Lorentz term
    eps_lorentz =wp**2 /((1j*(w0**2 - omega**2) - ((1j * omega) / tau)))
    eps_total = eps_inf + eps_drude + eps_lorentz
    
    print("{:.2e}".format(w0))
    print("{:.2e}".format(wp))
    return eps_total

def calculate_reflection_coefficient(dE, E_Ref):
    """
    Calculates the complex reflection coefficient from dE and E_Ref.
    """
    r_tilde = dE / E_Ref + 1
    return r_tilde

def fitfun(params, dE, E_Ref, freq):
    """
    Fit function to minimize the error between the measured and theoretical reflection coefficients.
    
    Parameters:
    - params: Fitting parameters [tau, wp].
    - dE: Measured delta E (difference in reflected field).
    - E_Ref: Reference reflected field.
    - freq: Frequency array.
    
    Returns:
    - Residual errors to minimize.
    """
    tau, wp = params
    sigma_drude = const.epsilon_0 * wp**2 * tau / (1 - 1j * (freq*w) * tau) 
    eps = calculate_eps(freq, tau, wp)
    n_static_real, n_static_imag  = unpumped_refractive_index("refractive_index_data_dec1.csv")
    n =  n_static_real+1j*n_static_imag
    # n = np.sqrt(eps)
    
    r_tilde_measured = calculate_reflection_coefficient(dE, E_Ref)
    r_tilde_theoretical = ((1 - n - Z0 * sigma_drude * d) /
                               (1 + n + Z0 * sigma_drude * d)) * ((1 + n) / (1 - n))
    # Calculate the residuals (error) between measured and theoretical coefficients
    y_diff = r_tilde_measured - r_tilde_theoretical
    err = (np.abs(r_tilde_measured) - np.abs(r_tilde_theoretical))**2 + (np.angle(r_tilde_measured) - np.angle(r_tilde_theoretical))**2
    #return y_diff.real**2 + y_diff.imag**2
    print(err)
    return err.flatten()

def fit_transient_reflection(dE, E_Ref, freq, init):
    """
    Fits the transient reflection coefficient data to extract tau and wp.
    
    Parameters:
    - dE: Measured delta E (difference in reflected field).
    - E_Ref: Reference reflected field.
    - freq: Frequency array.
    - initial_guess: Initial guess for [tau, wp].
    
    Returns:
    - Fitted parameters [tau, wp].
    """

    bounds = ([1e-14,  1e11], [1e-12, 1e14])
    result = least_squares(fitfun, init, bounds =bounds, args=(dE, E_Ref, freq) )
    tau_fit, wp_fit = result.x
    return tau_fit, wp_fit

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



def photoinduced_conductivity(sigma_drude, freqs, Z0, d, unpumped_refractive_index_file):
    n_real, n_imag  = unpumped_refractive_index(unpumped_refractive_index_file)
    n = n_real + 1j * n_imag
    r0 = (n-1)/(n+1)
    dr = 1-((1-r0)/r0)* (Z0*d*sigma_drude) / (1+n+Z0*d*sigma_drude)
    return dr, r0

import numpy as np

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


def conductivity_with_resonance(freq, omega_0, tau, wp):
    """
    Calculate the conductivity using the modified Drude model with a restoring force.

    Parameters:
    - freq: Frequency array (Hz).
    - N: Carrier density (m^-3).
    - tau: Relaxation time (s).
    - m_star: Effective mass of electron (kg).
    - omega_0: Resonance angular frequency (rad/s).

    Returns:
    - sigma: Complex conductivity (S/m).
    """
    omega = 2 * np.pi * freq  # Angular frequency
    sigma =  wp**2 * tau * const.epsilon_0/(1 - 1j * tau * (omega - (omega_0**2 / omega)))
    return sigma


if __name__ == "__main__":
    lockin1_file = "processed_data_roomtemp/LockIn1.csv"
    lockin2_file = "processed_data_roomtemp/LockIn2.csv"
    output_folder = "reflection_analysis_roomtemp"
    pump_delay = "0.4"
    plotting = True
    omega_0 = 2 * np.pi * 5e12  # Resonance frequency (rad/s), 5 THz

    E_Ref_windowed, E_Pump_windowed, dE, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column=pump_delay )
    R, E_Ref_FFT, E_Pump_FFT, freqs = analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis)
    initial_guess =  [1.00e-13,  6.8*1e13] # initial guess for tau and wp
    tau_fit, wp_fit = fit_transient_reflection(E_Pump_FFT-E_Ref_FFT, E_Ref_FFT,  freqs, initial_guess)
    print(f"Fitted tau: {tau_fit:.2e}, Fitted wp: {wp_fit:.2e}")
    n_static_real, n_static_imag = unpumped_refractive_index("refractive_index_data_dec1.csv")
    n_static =  n_static_real+1j*n_static_imag
    # Calculate theoretical reflectivity
    sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs) # The photoexcited part can be described by a Drude Lorentz model
    eps_drude = eps_inf - wp_fit**2 / ((w*freqs)**2 + 1j * (w*freqs) / tau_fit)
    # Lorentz term
    eps_lorentz = wp_fit**2 / (w0**2 - (w*freqs)**2 - 1j * (w*freqs) / tau_fit)

    eps_fit = eps_inf + eps_lorentz + eps_drude
    n_fit = np.sqrt(eps_fit)
    n_fit = n_static

    R_calculated = ((1 - n_fit - Z0 * sigma_drude_fit * d) /
                        (1 + n_fit + Z0 * sigma_drude_fit * d)) * ((1 + n_fit) / (1 - n_fit))
    R_experimental = calculate_reflection_coefficient(E_Pump_FFT-E_Ref_FFT, E_Ref_FFT)
    conductivity = calculate_sigma(R_experimental ,n_static)
    conductivity_lorentz = conductivity_with_resonance(freqs, omega_0, tau_fit, wp_fit)
    conductivity_total = conductivity_lorentz + sigma_drude_fit
    # Plot calculated and experimental reflectivity
    plt.figure(figsize=(10, 6))
   # plt.plot(freqs / 1e12, np.real(conductivity  ), label="Experimental Conductivity", linestyle="dotted")
   # plt.plot(freqs / 1e12, np.imag(conductivity ), label="Experimental Conductivity", linestyle="dotted")
    plt.plot(freqs / 1e12, np.real(conductivity_total), label="Fitted Conductivity")
    plt.plot(freqs / 1e12, np.imag(conductivity_total), label="Fitted Conductivity")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Reflectivity")
    plt.title(f"Reflectivity Comparison (Pump Delay = {pump_delay} ps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Plot calculated and experimental reflectivity
    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e12, np.real(R_experimental ), label="Real(R_exp)")
    plt.plot(freqs / 1e12, np.imag(R_experimental ), label="Imag(R_exp)")
    plt.plot(freqs / 1e12, np.real(R_calculated), label="Real(R_fit)", linestyle="--")
    plt.plot(freqs / 1e12, np.imag(R_calculated), label="Imag(R_fit)", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
