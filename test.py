import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.constants as const
from src.refractive import *
from src.conductivity import *

# Constants
Z0 = 376.73  # Free-space impedance (Ohms)
freq = np.linspace(0.1e12, 10e12, 100)  # Frequency in Hz (0.1 THz to 10 THz)

# Load refractive index data
n_static_real, n_static_imag = unpumped_refractive_index("refractive_index_data_dec1.csv")
n_static = n_static_real + 1j * n_static_imag

def update_plot(val):
    """Update the plot based on slider values."""
    tau = tau_slider.val
    wp = wp_slider.val
    eps_inf = eps_inf_slider.val
    gamma = gamma_slider.val
    d = d_slider.val

    # Calculate Drude-Lorentz conductivity
    sigma_drude = calculate_sigma_drude(tau, wp, freq)

    # Calculate reflectivity
    R_calculated = ((1 - n_static - Z0 * sigma_drude * d) /
                    (1 + n_static + Z0 * sigma_drude * d)) * ((1 + n_static) / (1 - n_static))

    # Update plot
    real_line.set_ydata(np.real(R_calculated))
    imag_line.set_ydata(np.imag(R_calculated))
    plt.draw()

# Initial parameters
d_init  = 5e-6
tau_init = 1.00e-12
wp_init = 9e13
eps_inf_init = 10.2
gamma_init = 1 / (2.80e-24)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initial calculations
sigma_drude = calculate_sigma_drude(tau_init, wp_init, freq)
R_calculated = ((1 - n_static - Z0 * sigma_drude * d_init) /
                (1 + n_static + Z0 * sigma_drude * d_init)) * ((1 + n_static) / (1 - n_static))

real_line, = ax.plot(freq / 1e12, np.real(R_calculated), label="Real(Reflectivity)")
imag_line, = ax.plot(freq / 1e12, np.imag(R_calculated), label="Imag(Reflectivity)")

ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Reflectivity")
ax.set_title("Reflectivity with Adjustable Parameters")
ax.legend()
ax.grid()

# Create sliders
ax_tau = plt.axes([0.1, 0.25, 0.8, 0.03])
ax_wp = plt.axes([0.1, 0.2, 0.8, 0.03])
ax_eps_inf = plt.axes([0.1, 0.15, 0.8, 0.03])
ax_gamma = plt.axes([0.1, 0.1, 0.8, 0.03])
ax_d = plt.axes([0.1, 0.05, 0.8, 0.03])

tau_slider = Slider(ax_tau, 'Tau (s)', 1e-13, 1e-11, valinit=tau_init, valstep=1e-12)
wp_slider = Slider(ax_wp, 'wp (rad/s)', 1e12, 1e15, valinit=wp_init, valstep=1e12)
eps_inf_slider = Slider(ax_eps_inf, 'Eps_Inf', 1, 20, valinit=eps_inf_init, valstep=0.1)
gamma_slider = Slider(ax_gamma, 'Gamma', 1e-25, 1e-23, valinit=gamma_init, valstep=1e-25)
d_slider = Slider(ax_d, 'd', 1e-7, 1e-4, valinit=d_init, valstep=5e-5)
# Connect sliders to update function
tau_slider.on_changed(update_plot)
wp_slider.on_changed(update_plot)
eps_inf_slider.on_changed(update_plot)
gamma_slider.on_changed(update_plot)
d_slider.on_changed(update_plot)

plt.show()
