import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from src.conductivity import *
from scipy.optimize import curve_fit


# Constants
d = 7.56e-7 # Penetration depth (m) (1micrometer)
w = 2 * np.pi   # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
tauD = 3.5e-12  # Debye relaxation time (s)
magD = 80  # Debye magnitude
#eps_inf = 8.601777326202962


def calculate_eps(freq, tau, wp, w0, eps_inf):
    eps_debye = magD / (1 - 1j * freq * w * tauD)
    eps_lorentz = 1j * wp**2 / (const.epsilon_0 * (1j * ((w0**2 - (w*freq)**2)) + w*freq /tau))
    eps_lorentz =  wp**2 / (const.epsilon_0 * ( ((w0**2 - (w*freq)**2)) +1j * w*freq /tau))
    return eps_inf + eps_debye + eps_lorentz



def calculate_transient_photoconductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, r0):
    E_pump = E_Pump_Amp * (np.cos(E_Pump_Phase) + 1j * np.sin(E_Pump_Phase))
    E_ref = E_Ref_Amp * (np.cos(E_Ref_Phase) + 1j * np.sin(E_Ref_Phase))
    delta_E = E_pump - E_ref
    delta_sigma = (-1 + r0) / Z0 * (delta_E / E_ref)
    return delta_sigma


def fitfun(params, R, eps_inf, freq):
    """
    Fit function to minimize the error between the measured and theoretical reflection coefficients.
    
    Parameters:
    - params: Fitting parameters [tau, wp].
    - R: complex reflection 
    - freq: Frequency array.
    
    Returns:
    - Residual errors to minimize.
    """
    tau, wp,  = params
    #n_static_real, n_static_imag = unpumped_refractive_index("refractive_index_data_dec1.csv")
    # Drude-Lorentz model for the static reflection function 
    wp_static = 1.79e+13 # rad/s
    gamma = 3.10e+12 
    wp_j = 5.81e+13 # Fitted Lorentz oscillator strength
    w0_j = 2.47e+13 
    gamma_j = 3.81e+12 
    omega_0 = w0_j
    n_static_real, n_static_imag = n_static_drude_lorentz(freq, eps_inf, wp_static, gamma, wp_j, w0_j, gamma_j)
    n_static =  n_static_real+1j*n_static_imag
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    sigma_drude  =  calculate_sigma_drude_lorentz(freq, omega_0,tau, wp) # #  Drude Lorentz model
    n_static =  n_static_real+1j*n_static_imag
    r_tilde_theoretical = ((1 - n_static - Z0 * sigma_drude * d) /
                               (1 + n_static + Z0 * sigma_drude * d)) * ((1 + n_static) / (1 - n_static))
    # Calculate the residuals (error) between measured and theoretical coefficients
    err = (np.abs(R) - np.abs(r_tilde_theoretical))**2 + (np.angle(R) - np.angle(r_tilde_theoretical))**2
    return err.flatten()


def fit_transient_reflection_2(R,eps_inf, freq, init):
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

    bounds = ([1e-15,  1e11], [1e-12, 1e14])
    #bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
    result = least_squares(fitfun, init, bounds =bounds, args=(R,  eps_inf, freq) )
    tau_fit, wp_fit = result.x
    return tau_fit, wp_fit



from lmfit import minimize, Parameters
from src.refractive import n_static_drude_lorentz

def reflection_model(params, freq):
    """
    Theoretical model for the transient reflection coefficient.

    Parameters:
    - params: lmfit Parameters object containing 'tau', 'wp', 'eps_inf', etc.
    - freq: Frequency array (Hz).

    Returns:
    - Complex reflection coefficient.
    """
    tau = params['tau']
    wp = params['wp']
    eps_inf = params['eps_inf']
    wp_static = params['wp_static']
    gamma = params['gamma']
    wp_j = params['wp_j']
    w0_j = params['w0_j']
    gamma_j = params['gamma_j']

    omega = 2 * np.pi * freq  # Angular frequency (rad/s)
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    sigma_drude  =  calculate_sigma_drude_lorentz(freq, w0_j, tau, wp) # #  Drude Lorentz model

    # Refractive index from Drude-Lorentz model
    n_static_real, n_static_imag = n_static_drude_lorentz(
        freq, eps_inf, wp_static, gamma, wp_j, w0_j, gamma_j
    )
    n_static = n_static_real + 1j * n_static_imag

    # Theoretical reflection coefficient
    r_tilde_theoretical = ((1 - n_static - Z0 * sigma_drude * d) /
                           (1 + n_static + Z0 * sigma_drude * d)) * ((1 + n_static) / (1 - n_static))
    # Offset Real and Imaginary part:
    r_tilde_theoretical = 1j*r_tilde_theoretical
    R_real = np.real(r_tilde_theoretical) + 1 
    R_imag = np.imag(r_tilde_theoretical) - 1
    r_tilde_theoretical  = R_real + 1j*R_imag
    return r_tilde_theoretical  

def residual(params, freq, R):
    """
    Residual function to minimize.

    Parameters:
    - params: lmfit Parameters object containing 'tau', 'wp', etc.
    - freq: Frequency array (Hz).
    - R: Measured reflection coefficient (complex array).

    Returns:
    - Residuals (real and imaginary parts concatenated).
    """
    r_tilde_theoretical = reflection_model(params, freq)
    residuals_real = np.real(R - r_tilde_theoretical)
    residuals_imag = np.imag(R - r_tilde_theoretical)
    return np.concatenate((residuals_real, residuals_imag))


def fit_transient_reflection(R, freq, init):
    """
    Fits the transient reflection coefficient data to extract tau and wp using lmfit.

    Parameters:
    - R: Measured reflection coefficient (complex array).
    - freq: Frequency array (Hz).
    - init: Dictionary of initial guesses for the parameters.

    Returns:
    - lmfit MinimizerResult object containing the fit results.
    """
    # Define parameters
    params = Parameters()
    params.add('tau', value=init['tau'], min=1e-15, max=1e-3)
    params.add('wp', value=init['wp'], min=1e13, max=1e15)
    params.add('eps_inf', value=8.601777326202962,  vary=False)
    params.add('wp_static', value=1.79e13, vary=False)  # Fixed parameter
    params.add('gamma', value=3.10e12, vary=False)  # Fixed parameter
    params.add('wp_j', value=5.81e13, vary=False)  # Fixed parameter
    params.add('w0_j', value=2.47e13, vary=False)  # Fixed parameter
    params.add('gamma_j', value=3.81e12, vary=False)  # Fixed parameter
    # Perform fit
    result = minimize(residual, params, args=(freq, R))
    return result
