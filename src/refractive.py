import pandas as pd
import numpy as np 

def unpumped_refractive_index(file_path):   
    """
    Assumes the input file is a CSV wfile which has the following columns: Frequency (Hz), Real Refractive Index, Imaginary Refractive Index

    """
    data = pd.read_csv(file_path)
    real_refractive_index = data['Real Refractive Index'].to_numpy()
    imaginary_refractive_index = data['Imaginary Refractive Index'].to_numpy()
    return real_refractive_index, imaginary_refractive_index


# Drude-Lorentz model for the static reflection function 
def n_static_drude_lorentz(freq, eps_inf, wp, gamma, wp_j, w0_j, gamma_j):
    """
    Combined Drude-Lorentz model for complex permittivity.
    
    Parameters:
    - freq: Frequency array (in Hz)
    - eps_inf: High-frequency dielectric constant
    - wp: Plasma frequency of free carriers (rad/s)
    - gamma: Damping constant of free carriers (rad/s)
    - wp_j: Oscillator strength of Lorentz term
    - w0_j: Resonance frequency of Lorentz term (rad/s)
    - gamma_j: Damping constant of Lorentz term (rad/s)
    
    Returns:
    - epsilon: Complex permittivity
    """
    omega = 2 * np.pi * freq
    # Drude term (ignore Drude term because CdTe is not conductive)
    eps_drude = wp**2 / (omega**2 + 1j * gamma * omega)
    # Lorentz term
    eps_lorentz = wp_j**2 / (w0_j**2 - omega**2 - 1j * gamma_j * omega)
    # Total permittivity
    epsilon = eps_inf + eps_lorentz + eps_drude
    n = np.sqrt(epsilon)
    return np.real(n),np.imag(n) 