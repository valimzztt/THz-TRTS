import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.constants as const

# Define the Drude model for conductivity 
def drude_conductivity(freq, wp, gamma):
    """
    Drude model for the complex permittivity.

    Parameters:
    - freq (array): Frequency in Hz.
    - eps_inf (float): High-frequency permittivity.
    - wp (float): Plasma frequency (rad/s).
    - gamma (float): Scattering rate (rad/s).

    Returns:
    - sigma_real (array): Real part of conductivity.
    - sigma_imag (array): Imaginary part of conductivity.
    """
    omega = 2 * np.pi * freq
    tau = 1/gamma
    #eps_drude = eps_inf - (wp**2 / (omega**2 + 1j * omega * gamma))
    sigma_drude = (wp**2 * tau*const.epsilon_0)/(1 - 1j * omega * tau)
    return sigma_drude.real, sigma_drude.imag

def drude_conductivity1(freq):
    """
    Drude model for the complex permittivity.

    Parameters:
    - freq (array): Frequency in Hz.
    - eps_inf (float): High-frequency permittivity.
    - wp (float): Plasma frequency (rad/s).
    - gamma (float): Scattering rate (rad/s).

    Returns:
    - eps_real (array): Real part of conductivity.
    - eps_imag (array): Imaginary part of conductivity.
    """
    tau = 1e-13  # Relaxation time, tau = 100fs
    gamma = 1/tau 
    N = 1e23  # Carrier density, N = 1e17 cm^-3 = 1e23 m^-3
    m  = 0.067*9.11e-31  # Effective mass of electron
    q = 1.6e-19  # Charge of electron
    omega = 2 * np.pi * freq
    wp = np.sqrt(N * q**2 / (m * const.epsilon_0))
    print("Plasma frequency: ", wp/1e12)
    eps_drude = (N*q**2)*tau/(m*(1- 1j*tau*omega))
    eps_drude = (wp**2 * tau*const.epsilon_0)/(1 - 1j * omega * tau)
    return eps_drude.real, eps_drude.imag

wp  = 68*1e12
eps_inf = 10.2
gamma = 1e11
freq = np.linspace(0.1e12, 10e12, 1000)  # Frequency in Hz (0.1 THz to 5 THz)
omega = 2 * np.pi *freq 
eps_drude =  (wp**2 / (omega**2 + 1j * omega * gamma)) 

eps_real_exp = eps_drude.real  
eps_imag_exp = eps_drude.imag  
# Combine real and imaginary parts into a single array for fitting
eps_real_exp, eps_imag_exp = drude_conductivity1(freq)
eps_exp = np.concatenate([eps_real_exp, eps_imag_exp])

# Fit function: Real and imaginary parts combined
def fit_function(freq, wp, gamma):
    eps_real, eps_imag = drude_conductivity(freq,wp, gamma)
    return np.concatenate([eps_real, eps_imag])

# Initial guesses for fitting parameters
p0 = [68*1e12, 1e11] # wp, gamma
bounds =  ([0, 0], [np.inf, np.inf])
# Perform the fit
params, _ = curve_fit(fit_function, freq, eps_exp, p0=p0, bounds=bounds)

# Extract fitted parameters
wp_fit, gamma_fit = params
print(f"Fitted Parameters:\n   ωp = {wp_fit:.2e} rad/s\n  γ = {gamma_fit:.2e} rad/s  tau = {1/gamma_fit:.2e} s")

# Generate fitted permittivity for comparison
eps_real_fit, eps_imag_fit = drude_conductivity(freq, wp_fit, gamma_fit)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(np.log(freq), eps_real_fit, 'o', label="Real(ε) Fit")
plt.plot(np.log(freq), eps_imag_fit, 'o', label="Imag(ε) Fit")
#plt.plot(np.log(freq), eps_real_fit, '-', label="Real(ε) Fitted")
#plt.plot(freq / 1e12, eps_imag_fit, '-', label="Imag(ε) Fitted")
plt.xlabel("Frequency (THz)")
plt.ylabel("Permittivity")
plt.legend()
plt.grid()
plt.title("Fit to Drude Model")
plt.show()
plt.close()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(np.log(freq), eps_real_exp, 'o', label="Real(ε) Experimental")
plt.plot(np.log(freq), eps_imag_exp, 'o', label="Imag(ε) Experimental")
#plt.plot(np.log(freq), eps_real_fit, '-', label="Real(ε) Fitted")
#plt.plot(freq / 1e12, eps_imag_fit, '-', label="Imag(ε) Fitted")
plt.xlabel("Frequency (THz)")
plt.ylabel("Permittivity")
plt.legend()
plt.grid()
plt.title("Experimental Data")
plt.show()
