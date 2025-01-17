# Scripts that computes the skin depth for a given frequency and refractive index.
import numpy as np
import scipy.constants as const 
def skin_depth(freq, mu_r, sigma):
    """
    Calculate the skin depth for a given frequency, refractive index and penetration depth.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    mu_r : float
        Relative permeability.
    
    d : float
        Real part of the conductivity in S/m.

    Returns
    -------
    float
        Skin depth in meters.
    """
    return 1 / np.sqrt(np.pi * freq * mu_r * const.mu_0* sigma)



def skin_depth_2(freq, mu_r, rho):
    """
    Calculate the skin depth for a given frequency, refractive index and penetration depth.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    mu_r : float
        Relative permeability.
    rho : float
        Resitivity

    Returns
    -------
    float
        Skin depth in meters.
    """
    return np.sqrt(rho/(np.pi*freq*mu_r*const.mu_0))
                             
# For CdTe at 800 nm
freq_nm = 800
freq_GHz = 299792458/freq_nm
freq_Hz = freq_GHz*1e12
mu = 8.6531 
sigma = 3.72*1e-4 #(Ωm)-1
resitivity = 1/sigma # Ωm
resitivity = resitivity *1e8 # micro-ohm cm
print(f"Resistivity: {resitivity:.2e} Ωm")
d = skin_depth(freq_Hz, mu, sigma)
#d = skin_depth_2(freq_GHz, mu, resitivity)

print(f"The skin depth for CdTe at {freq_nm} nm is {d:.2e} m.")