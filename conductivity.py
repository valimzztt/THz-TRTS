import scipy.constants as const 
import numpy as np
import matplotlib.pyplot as plt

def drude_conductivity(freq):
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
    
    eps_drude = (wp**2 * tau*const.epsilon_0)/(1 - 1j * omega * tau)
    return eps_drude

def drude_smith_conductivity(freq):
    """
    Drude-Smith model for the complex conductivity.

    Parameters:
    - freq (array): Frequency in Hz.

    Returns:
    - eps_drude_smith (array): Complex conductivity for the Drude-Smith model.
    """
    tau = 1e-13  # Relaxation time, tau = 100 fs
    N = 1e23  # Carrier density, N = 1e17 cm^-3 = 1e23 m^-3
    m = 0.067 * 9.11e-31  # Effective mass of electron
    q = 1.6e-19  # Charge of electron
    omega = 2 * np.pi * freq
    c = 0.3  # Drude-Smith backscattering parameter
    wp = np.sqrt(N * q**2 / (m * const.epsilon_0))
    sigma_drude = wp**2 * tau * const.epsilon_0 / (1 - 1j * omega * tau)
    correction_term = c / (1 - 1j * omega * tau)
    sigma_drude_smith = sigma_drude * (1 + correction_term)
    return sigma_drude_smith


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


import numpy as np
import scipy.constants as const

def calculate_dielectric_function(freq, eps_lattice, sigma):
    """
    Calculates the dielectric function from the given parameters.

    Parameters:
    - freq: Frequency array in Hz.
    - eps_lattice: Lattice dielectric constant (epsilon_L).
    - sigma: Complex conductivity array (sigma(omega)).

    Returns:
    - eps: Complex dielectric function array.
    """
    omega = 2 * np.pi * freq  # Angular frequency
    eps0 = const.epsilon_0  # Permittivity of free space (F/m)

    # Dielectric function
    eps = eps_lattice + 1j * sigma / (omega * eps0)

    return eps

# Example usage
freqs = np.linspace(1e12, 10e12, 1000)  # Frequency range from 1 THz to 10 THz
eps_lattice = 10.4  # Example lattice dielectric constant
sigma = 1e3 + 1j * 5e2  # Example complex conductivity

# Calculate the dielectric function
dielectric_function = calculate_dielectric_function(freqs, eps_lattice, sigma)




freqs = np.linspace(0.1e12, 10e12, 1000)  # Frequency in Hz (0.1 THz to 5 THz)
tau = 1e-13  # Relaxation time, tau = 100fs
N = 1e23  # Carrier density, N = 1e17 cm^-3 = 1e23 m^-3
m  = 0.067*9.11e-31  # Effective mass of electron
q = 1.6e-19  # Charge of electron
omega = 2 * np.pi * freqs
wp = np.sqrt(N * q**2 / (m * const.epsilon_0))

conductivity_drude = drude_conductivity(freqs)
conductivity_drude_smith = drude_smith_conductivity(freqs)
omega_0 = 2 * np.pi * 5e12  # Resonance frequency (rad/s), assuming 5 THz

# Calculate conductivity
sigma = conductivity_with_resonance(freqs,omega_0, tau, wp)
sigma_total = conductivity_drude  +  sigma
plt.figure(figsize=(10, 6))

plt.plot(freqs / 1e12, np.real(conductivity_drude), label="Real(sigma_drude)", linestyle="--")
plt.plot(freqs / 1e12, np.imag(conductivity_drude), label="Imag(sigma_drude)", linestyle="--")
plt.plot(freqs / 1e12, np.real(conductivity_drude_smith), label="Real(sigma_drude_smith)", linestyle="--")
plt.plot(freqs / 1e12, np.imag(conductivity_drude_smith), label="Imag(sigma_drude_smith)", linestyle="--")
""" plt.plot(freqs/1e12, np.real(sigma), label="Real(sigma_lorentz)", linestyle="-")
plt.plot(freqs/1e12, np.imag(sigma), label="Imag(sigma_lorentz)", linestyle="-") """
plt.plot(freqs/1e12, np.real(sigma_total), label="Real(sigma_tot)", linestyle="-")
plt.plot(freqs/1e12, np.imag(sigma_total), label="Real(sigma_tot)", linestyle="-")
plt.xlabel("Frequency (THz)")
plt.ylabel("Reflectivity")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig(f"{output_folder}/reflectivity_comparison_{pump_delay}.png")
plt.show()

# Plot the real and imaginary parts
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(freqs / 1e12, np.real(dielectric_function), label="Real(ε(ω))")
plt.plot(freqs / 1e12, np.imag(dielectric_function), label="Imag(ε(ω))")
plt.xlabel("Frequency (THz)")
plt.ylabel("Dielectric Function (ε(ω))")
plt.title("Dielectric Function vs Frequency")
plt.legend()
plt.grid(True)
plt.show()

ù