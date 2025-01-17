import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
Z0 = 377  # Impedance of free space (Ohms)

# Material parameters for GaAs
res_freq_THz = 5 
omega_0 = res_freq_THz*(2*np.pi)*1e12  # Resonance frequency (rad/s)
gamma = 1e11  # Damping constant (rad/s)
omega_pL = 6.88e+13  # Plasma frequency for Lorentz term (rad/s)
tau = 1.00e-13 # Drude relaxation time (s)
omega_pD =  6.88e+13  # Plasma frequency for Drude term (rad/s)
eps_inf = 10.9  # High-frequency permittivity
d = 1e-6
# Frequency range (THz -> rad/s)
freqs = np.linspace(0.1e12, 10e12, 1000)  # Frequency in Hz
omega = 2 * np.pi * freqs  # Convert to rad/s

# Drude term
sigma_drude = eps0 * omega_pD**2 * tau / (1 - 1j * omega * tau)
eps_drude = eps_inf - omega_pD**2 / (omega**2 + 1j * omega / tau)
# Lorentz term
eps_lorentz = omega_pL**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

eps_total = eps_inf + eps_lorentz + eps_drude 
sigma_total = -1j * omega * eps0 * eps_total  
n = np.sqrt(eps_total)
reflectivity = r_tilde_theoretical = 1 + ((1 - n - Z0 * sigma_drude * d) /
                               (1 + n + Z0 * sigma_drude * d)) * ((1 + n) / (1 - n))

# Plot results
plt.figure(figsize=(12, 8))

# Plot dielectric function (real and imaginary parts)
plt.subplot(3, 1, 1)
plt.plot(np.log(freqs), np.real(sigma_drude), label="Real(\u03B5)")
plt.plot(np.log(freqs), np.imag(sigma_drude), label="Imag(\u03B5)")
plt.title("Dielectric Function (GaAs)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Dielectric Function")
plt.legend()
plt.grid(True)

# Plot conductivity (real and imaginary parts)
plt.subplot(3, 1, 2)
plt.plot(freqs / 1e12, np.real(sigma_total), label="Real(\u03C3)")
plt.plot(freqs / 1e12, np.imag(sigma_total), label="Imag(\u03C3)")
plt.title("Conductivity (GaAs)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Conductivity (S/m)")
plt.legend()
plt.grid(True)

# Plot reflectivity (real and imaginary parts)
plt.subplot(3, 1, 3)
plt.plot(freqs / 1e12, np.real(reflectivity), label="Real(\u03C3)")
plt.plot(freqs / 1e12, np.imag(reflectivity), label="Imag(\u03C3)")
plt.title("Conductivity (GaAs)")
plt.xlabel("Frequency (THz)")
plt.ylabel("Conductivity (S/m)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
