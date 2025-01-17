import scipy.constants as const
import numpy as np 
from src.refractive import *

def calculate_sigma_drude(tau, wp, freq):
    """
    Calculate the Drude model conductivity for free carriers in a material.

    Parameters:
        tau (float): Relaxation time of the free carriers (in seconds).
        wp (float): Plasma frequency (in radians per second).
        freq (numpy.ndarray): Frequency array (in Hz).

    Returns:
        numpy.ndarray: Complex Drude conductivity as a function of frequency.
    """
    omega = 2 * np.pi * freq  # Convert frequency from Hz to angular frequency (rad/s)
    return const.epsilon_0 * tau * wp**2 / (1 - 1j * tau * omega)
def calculate_sigma_drude_lorentz(freq, omega_0, tau, wp):
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
    sigma =  const.epsilon_0 * tau * wp**2/(1 - 1j * tau * (omega - (omega_0**2 / omega)))
    return sigma


def sigma_dark(freq, eps, epsi):
    """
    Calculate the dark conductivity of a material based on its permittivity.

    Parameters:
        freq (numpy.ndarray): Frequency array (in Hz).
        eps (float): Photoexicted permittivity
        epsi (float): Non-photoexcited permittivity.
    """
    omega = 2 * np.pi * freq  # Convert frequency from Hz to angular frequency (rad/s)
    sigma_dark = (1j * const.epsilon_0 * omega) * (epsi - eps)  # Calculate dark conductivity
    return sigma_dark

def photoinduced_conductivity(sigma_drude,  Z0, d, n_real, n_imag):
    n = n_real + 1j * n_imag
    r0 = (n-1)/(n+1)
    dr = 1-((1-r0)/r0)* (Z0*d*sigma_drude) / (1+n+Z0*d*sigma_drude)
    return dr, r0


def free_carrier(wp):
    nfree = const.epsilon_0*(0.2*const.electron_mass)*wp**2/(const.e^2)
    return nfree

