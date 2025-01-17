import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.constants as const
from src.refractive import *
from src.utils import *
from src.conductivity import *
from src.fitting_roomtemp import *
from src.reflectivity import *

# Define constants
Z0 = 376.73  # Impedance of free space (Ohms)
d = 7.56e-7 # Penetration depth (m) (1micrometer)
# Define the callback class
class Index:
    def __init__(self):
        self.ind = 10
        self.selection = "Reflectivity"

    def next(self, _):
        self.ind += 1
        if self.ind >= len(pump_delays):
            self.ind = 0
        update_plot()

    def prev(self, _):
        self.ind -= 1
        if self.ind < 0:
            self.ind = len(pump_delays) - 1
        update_plot()

    def toggle_selection(self, _):
        options = ["Reflectivity", "Reflectance", "Raw VS Windowed Pulse" , "Fit Conductivity","Fit Dielectric", "Photo-induced Conductivity", "E_Ref & E_Pump"]
        current_index = options.index(self.selection)
        self.selection = options[(current_index + 1) % len(options)]
        update_plot()

# Define the update plot function
def update_plot():
    axs[0].cla()
    axs[1].cla()

    pump_delay = pump_delays[callback.ind]
    E_Ref, E_Pump, dE,  time_axis = read_data(lockin1_file, lockin2_file, time_delay_column=pump_delay)
    E_Ref, E_Pump, dE,  time_axis =  extend_to_power_of_2(E_Ref, E_Pump, dE, time_axis)
    E_Ref_FFT_raw_abs, E_Pump_FFT_raw_abs, freqs_raw = fft(E_Ref, E_Pump, time_axis)
    E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column=pump_delay)
    E_Ref_windowed, E_Pump_windowed, dE_windowed,  time_axis =  extend_to_power_of_2(E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis)
    R, E_Ref_FFT, E_Pump_FFT, freqs = analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis)
    E_Ref_FFT_windowed_abs, E_Pump_FFT_windowed_abs, freqs_windowed = fft(E_Ref_windowed, E_Pump_windowed, time_axis)
    initial_guess =  [1.00e-12,  6.8*1e13] # Initial guess for tau and wp
    R_exp = calculate_reflection_coefficient( E_Pump_FFT-E_Ref_FFT, E_Ref_FFT)
    reflectivity, _  = calculate_reflectivity(E_Ref, E_Pump, time_axis)
    dr, _  = calculate_differential_reflectivity(E_Ref, E_Pump, time_axis)
    # Initial guesses
    init = {
        'tau': 1e-13,
        'wp': 9e13,
    }

    # Perform the fit
    fit_result  = fit_transient_reflection(R, freqs, init)
    # Plot results
    R_fitted = reflection_model(fit_result.params, freqs)
    tau_fit = fit_result.params['tau'].value
    wp_fit = fit_result.params['wp'].value
    eps_inf_fit = fit_result.params['eps_inf'].value
    print(f"Pump delay: {pump_delay} ps")
    print(f"Fitted tau: {tau_fit:.2e} s")
    print(f"Fitted wp: {wp_fit:.2e} rad/s")


    if not hasattr(callback, 'results_list'):
            callback.results_list = []

    callback.results_list.append({
        "Pump Delay (ps)": pump_delay,
        "Fitted Tau (s)": f"{tau_fit:.2e}",
        "Fitted wp (rad/s)": f"{wp_fit:.2e}"
    })

    # Drude-Lorentz model for the static reflection function 
    wp_static = 1.79e+13 # rad/s
    gamma = 3.10e+12 
    wp_j = 5.81e+13 # Fitted Lorentz oscillator strength
    w0_j = 2.47e+13 
    gamma_j = 3.81e+12 
    n_static_real, n_static_imag = n_static_drude_lorentz(freqs, eps_inf, wp_static, gamma, wp_j, w0_j, gamma_j)
    n_static =  n_static_real+1j*n_static_imag
    omega_0 = w0_j 
    # Calculate theoretical reflectivity from the fitted Plasma frequency and Tau 
    sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs) # The photoexcited part can be described by a Drude Lorentz model
    sigma_drude_lorentz_fit =  calculate_sigma_drude_lorentz(freqs, omega_0,tau_fit, wp_fit) # #  Drude Lorentz model
    R_calculated = ((1 - n_static - Z0 * sigma_drude_fit * d) /
                    (1 + n_static + Z0 * sigma_drude_fit * d)) * ((1 + n_static) / (1 - n_static))
    # Modify calculated Reflection to match calculated one 
    R_calculated = 1j*R_calculated 
    R_real = np.real(R_calculated ) + 1
    R_imag = np.imag(R_calculated ) - 1
    R_calculated = R_real + 1j*R_imag
    # Fitted Sigma Drude Fit
    # Photo-induced conductivity  "Raw Puse" , "FFT"
    eps =  calculate_eps(freqs, tau_fit, wp_fit, omega_0, eps_inf)
    n = np.sqrt(eps)
    n_real = np.real(n)
    n_imag = np.imag(n)
    transient_photoconductivity, r0 = photoinduced_conductivity(sigma_drude_fit,  Z0, d, n_real, n_imag)

    if callback.selection == "Raw VS Windowed Pulse":
        axs[0].set_title(f"Pulse in time, pump delay= {pump_delay} ps")
        axs[0].plot(time_axis, E_Ref, label="E_Ref (raw)")
        axs[0].plot(time_axis, dE, label="dE (raw)")
        axs[0].plot(time_axis, E_Ref_windowed, label="E_Ref (windowed)")
        axs[0].plot(time_axis, dE_windowed, label="dE (windowed)")
        axs[0].set_xlabel("Time (ps)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()
        axs[0].set_xlim(0, 7)
        axs[0].grid(True)

        axs[1].set_title(f"FFT, pump delay= {pump_delay} ps")
        axs[1].plot(freqs_raw , E_Ref_FFT_raw_abs, label="FFT(E_Ref_raw)")
        axs[1].plot(freqs_windowed, E_Ref_FFT_windowed_abs, label="FFT(E_Ref_window)")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Intensity (a.u.)")
        axs[1].set_xlim(0, 20)
        axs[1].legend()
        axs[1].grid(True)
    
    if callback.selection == "E_Ref & E_Pump":
        axs[0].set_title(f"Pulse in time, pump delay= {pump_delay} ps")
        axs[0].plot(time_axis, E_Ref_windowed, label="E_Ref (windowed)")
        axs[0].plot(time_axis, dE_windowed, label="dE (windowed)")
        axs[0].set_xlabel("Time (ps)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()
        axs[0].set_xlim(0, 8)
        axs[0].grid(True)

        axs[1].set_title(f"Reflectance, pump delay= {pump_delay} ps")
        axs[1].plot(time_axis, E_Ref_windowed, label="E_Ref (windowed)")
        axs[1].plot(time_axis, E_Pump_windowed, label="E_Pump (windowed)")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Reflectivity")
        axs[1].legend()
        axs[1].grid(True)

    if callback.selection == "Reflectance":
        axs[0].set_title(f"Pulse in time, pump delay= {pump_delay} ps")
        axs[0].plot(time_axis, E_Ref_windowed, label="E_Ref")
        axs[0].plot(time_axis, dE_windowed, label="dE")
        axs[0].set_xlabel("Time (ps)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()
        axs[0].set_xlim(0, 8)
        axs[0].grid(True)

        axs[1].set_title(f"Reflectance, pump delay= {pump_delay} ps")
        axs[1].plot(freqs / 1e12, np.abs(R) - 1, label="Abs(R)-1")
        axs[1].plot(freqs / 1e12, np.abs(R_fitted) - 1, label="Abs(R_fitted)-1")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Reflectivity")
        axs[1].legend()
        axs[1].grid(True)
    
    if callback.selection == "Reflectivity":
        axs[0].set_title(f"Pulse in time, pump delay= {pump_delay} ps")
        axs[0].plot(time_axis, E_Ref_windowed, label="E_Ref")
        axs[0].plot(time_axis, dE_windowed, label="dE")
        axs[0].set_xlabel("Time (ps)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()
        axs[0].set_xlim(0, 8)
        axs[0].grid(True)

        axs[1].set_title(f"Reflectivity, pump delay= {pump_delay} ps")
        axs[1].plot(freqs / 1e12, np.real(R), label="Real(R_exp)")
        axs[1].plot(freqs / 1e12, np.imag(R), label="Imag(R_exp)") 
        axs[1].plot(freqs / 1e12, np.real(R_calculated), label="Real(R_fit)", linestyle="--")
        axs[1].plot(freqs / 1e12, np.imag(R_calculated), label="Imag(R_fit)", linestyle="--")
        axs[1].plot(freqs / 1e12, np.real(dr), label="Real(dr)")
        axs[1].plot(freqs / 1e12, np.imag(dr), label="Imag(dr") 
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Intensity (a.u.)")
        axs[1].legend()
        axs[1].set_xlim(0, 10)
        axs[1].grid(True)

    elif callback.selection == "Fit Conductivity":
        axs[0].set_title(f"Drude Conductivity, pump delay= {pump_delay} ps")
        axs[0].plot(np.log(freqs / 1e12), np.real(sigma_drude_fit), label="Real(sigma_D)")
        axs[0].plot(np.log(freqs / 1e12), np.imag(sigma_drude_fit), label="Imag(sigma_D)")
        axs[0].set_xlabel("Frequency (THz)")
        axs[0].set_ylabel("Conductivity (S/m)")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_xlim(0, 8)
        axs[0].grid(True)

        axs[1].set_title(f"Drude-Lorentz Conductivity Fit pump delay= {pump_delay} ps")
        axs[1].plot(freqs / 1e12, np.real(sigma_drude_lorentz_fit), label="Real(sigma_DL)")
        axs[1].plot(freqs / 1e12, np.imag(sigma_drude_lorentz_fit), label="Imag(sigma_DL)")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Conductivity (S/m)")
        axs[1].legend()
        axs[1].grid(True)
    elif callback.selection == "Fit Dielectric":
        axs[0].set_title(f"Dielectric function fit, pump delay= {pump_delay} ps")
        axs[0].plot(freqs / 1e12, np.real(eps), label="Real(eps)")
        axs[0].plot(freqs / 1e12, np.imag(eps), label="Imag(eps)")
        axs[0].set_xlabel("Frequency (THz)")
        axs[0].set_ylabel("Conductivity (S/m)")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_xlim(0, 8)
        axs[0].grid(True)

        axs[1].set_title(f"Drude-Lorentz Conductivity Fit pump delay= {pump_delay} ps")
        axs[1].plot(freqs / 1e12, np.real(sigma_drude_lorentz_fit), label="Real(sigma_DL)")
        axs[1].plot(freqs / 1e12, np.imag(sigma_drude_lorentz_fit), label="Imag(sigma_DL)")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Intensitiy(a.u)")
        axs[1].legend()
        axs[1].grid(True)
    elif callback.selection == "Photo-induced Conductivity":
        axs[0].set_title(f"Reflectivity, pump delay= {pump_delay} ps")
        axs[0].plot(freqs / 1e12, np.real(R_exp), label="Real(R_exp)")
        axs[0].plot(freqs / 1e12, np.imag(R_exp), label="Imag(R_exp)") 
        axs[0].set_xlabel("Frequency (THz)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()
        axs[0].set_xlim(0, 10)
        axs[0].grid(True)
        axs[1].set_title(f"Transient Photoconductivity, pump delay= {pump_delay} ps")
        axs[1].plot(freqs / 1e12, np.real(transient_photoconductivity), label="Real(delta_sigma)")
        axs[1].plot(freqs / 1e12, np.imag(transient_photoconductivity), label="Imag(delta_sigma)")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Photoconductivity (S/m)")
        axs[1].legend()
        axs[1].grid(True)
    plt.draw()

# Main script
if __name__ == "__main__":
    lockin1_file = "processed_data_roomtemp/LockIn1.csv"
    lockin2_file = "processed_data_roomtemp/LockIn2.csv"
    output_folder = "reflection_analysis_roomtemp"
    lockin1_data = pd.read_csv(lockin1_file)
    pump_delays = lockin1_data.columns[1:].tolist()
    # Get the static refractive function
    eps_inf = 8.601777326202962 # from THz TDS

    fig, axs = plt.subplots(1, 2, figsize=(12, 7))

    callback = Index()
    axprev = plt.axes([0.7, -0.01, 0.1, 0.045])
    axnext = plt.axes([0.81, -0.01, 0.1, 0.045])
    axselect = plt.axes([0.59,-0.01, 0.1, 0.045])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    bselect = Button(axselect, 'Selection')
    bselect.on_clicked(callback.toggle_selection)

    update_plot()
    plt.show()
