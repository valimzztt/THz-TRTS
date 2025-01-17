import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.constants as const
from src.refractive import *
from src.utils import *
from src.conductivity import *
from src.fitting_roomtemp import *


Z0 = 376.73 
d = 1.3e-6  # Sample thickness (m)
wp  = 9*1e13
tau = 1.00e-12 

eps_inf = 10.2
gamma = 1/(2.80e-24)
freq = np.linspace(0.1e12, 10e12, 100)  # Frequency in Hz (0.1 THz to 5 THz)
""" n_static_real, n_static_imag = unpumped_refractive_index("refractive_index_data_dec1.csv")
n_static =  n_static_real+1j*n_static_imag  """
# Drude-Lorentz model for the static reflection function 
wp_static = 1.79e+13 # rad/s
gamma = 3.10e+12 
wp_j = 5.81e+13 # Fitted Lorentz oscillator strength
w0_j = 2.47e+13 
gamma_j = 3.81e+12 
n_static_real, n_static_imag = n_static_drude_lorentz(freq, eps_inf, wp_static, gamma, wp_j, w0_j, gamma_j)
n_static =  n_static_real+1j*n_static_imag
omega_0 = w0_j
omega = 2 * np.pi * freq  # Convert frequency from Hz to angular frequency (rad/s)
sigma_drude  = const.epsilon_0 * tau * wp**2 / (1 - 1j * tau * omega)
sigma_drude_lorentz =  const.epsilon_0 * tau * wp**2/(1 - 1j * tau * (omega - (omega_0**2 / omega)))
R_calculated = ((1 - n_static - Z0 * sigma_drude_lorentz * d) /
                        (1 + n_static + Z0 * sigma_drude_lorentz * d)) * ((1 + n_static) / (1 - n_static))


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(freq/1e12, np.real(R_calculated) ,'o', label="Real(ε) Fit")
plt.plot(freq/1e12,  np.imag(R_calculated), 'o', label="Imag(ε) Fit")
plt.xlabel("Frequency (THz)")
plt.ylabel("Permittivity")
plt.legend()
plt.grid()
plt.title("Fitted Reflectivity")
plt.show()