import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Button
import scipy.constants as const

# Constants
d = 1.3e-6  # Penetration depth (m)
w = 2 * np.pi   # Angular frequency for 1 THz
Z0 = 376.73  # Impedance of free space (Ohms)
e = 1.602176565e-19  # Electron charge (C)
m = 9.10938291e-31  # Electron mass (kg)
eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
epsi = 8.6531 # Static permittivity for CdTe (Real part of refractive index)
gam = 3.61e+13   # Damping constant (Hz)
w0 = 5.0 * w  # Resonance frequency (rad/s)
wpL = 1.0e14 # Plasma frequency (rad/s)
tau = 1.0e-13  # Relaxation time (s)
tauD = 3.5e-12  # Debye relaxation time (s)
magD = 80  # Debye magnitude


def calculate_sigma_drude(tau, wp, freq):
    omega = 2 * np.pi * freq
    return const.epsilon_0 * tau * wp**2 / (1 - 1j * tau * omega)


def calculate_eps(freq, tau, wp):
    eps_debye = magD / (1 - 1j * freq * w * tauD)
    eps_lorentz = 1j * wpL**2 / (eps0 * (1j * ((w0**2 - (w*freq)**2)) + w*freq * gam))
    # return epsi + eps_debye + eps_lorentz
    return epsi  + eps_lorentz # Let's ignore the Debye term for now


def calculate_reflectivity(n, sigma_drude):
    r0 = (1 - n) / (1 + n)
    Re = -(1+r0)/r0 * ( Z0*d*sigma_drude ) / ( 1 + n + Z0*d*sigma_drude ) + 1
    return r0, Re

def calculate_transient_photoconductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, r0):
    E_pump = E_Pump_Amp * (np.cos(E_Pump_Phase) + 1j * np.sin(E_Pump_Phase))
    E_ref = E_Ref_Amp * (np.cos(E_Ref_Phase) + 1j * np.sin(E_Ref_Phase))
    delta_E = E_pump - E_ref
    delta_sigma = (-1 + r0) / Z0 * (delta_E / E_ref)
    return delta_sigma

def fitfun(params, R_measured, freq):
    tau, wp = params
    sigma_drude = calculate_sigma_drude(tau, wp, freq)
    eps = calculate_eps(freq, tau, wp)
    n = np.sqrt(eps)
    R_calculated = (1-n-Z0*d*sigma_drude) / (1+n+Z0*d*sigma_drude) / ( (1-n)/(1+n) )
    error = np.abs(R_measured - R_calculated)**2 + (np.angle(R_measured) - np.angle(R_calculated))**2
    return error.flatten()

def unpumped_refractive_index(file_path):
    data = pd.read_csv(file_path)
    real_refractive_index = data['Real Refractive Index']
    imaginary_refractive_index = data['Imaginary Refractive Index']
    return real_refractive_index, imaginary_refractive_index

def photoinduced_conductivity(sigma_drude, freqs, Z0, d, unpumped_refractive_index_file):
    n_real, n_imag  = unpumped_refractive_index(unpumped_refractive_index_file)
    n = n_real + 1j * n_imag
    r0 = (n-1)/(n+1)
    dr = 1-((1-r0)/r0)* (Z0*d*sigma_drude) / (1+n+Z0*d*sigma_drude)
    return dr, r0


def process_data(lockin1_file, lockin2_file, time_delay_column):
    lockin1_data = pd.read_csv(lockin1_file)
    lockin2_data = pd.read_csv(lockin2_file)
    global time_axis
    time_axis = lockin1_data.iloc[:, 0].values
    lockin1_signals = lockin1_data[time_delay_column].values
    lockin2_signals = lockin2_data[time_delay_column].values
    lockin1_signals -= np.mean(lockin1_signals)
    lockin2_signals -= np.mean(lockin2_signals)
    E_Ref = lockin1_signals - lockin2_signals
    E_Pump = lockin1_signals + lockin2_signals
    global processed_E_Ref, processed_E_Pump
    global E_Ref_windowed, E_Pump_windowed

    def create_centered_hann(data_length, window_center, window_width):
        hann_window = np.hanning(window_width)
        padded_window = np.zeros(data_length)
        start = max(0, window_center - window_width // 2)
        end = min(data_length, start + window_width)
        padded_window[start:end] = hann_window[:(end - start)]
        return padded_window

    hann_window_center = 100
    hann_window_width = 400
    ref_hann_window = create_centered_hann(len(E_Ref), hann_window_center, hann_window_width)
    pump_hann_window = create_centered_hann(len(E_Pump), hann_window_center, hann_window_width)
    E_Ref_windowed = E_Ref * ref_hann_window
    E_Pump_windowed = E_Pump * pump_hann_window

    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]
    E_Ref_FFT = np.fft.fft(E_Ref)[:len(freqs)]
    E_Pump_FFT = np.fft.fft(E_Pump)[:len(freqs)]
    E_Ref_windowed_FFT = np.fft.fft(E_Ref_windowed)[:len(freqs)]
    E_Pump_windowed_FFT = np.fft.fft(E_Pump_windowed)[:len(freqs)]

    return E_Ref_windowed, E_Pump_windowed, time_axis


def analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis):
    E_Ref = E_Ref_windowed
    E_Pump = E_Pump_windowed
    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]

    E_Ref_Amp = np.abs(np.fft.fft(E_Ref))[:len(freqs)]
    E_Pump_Amp = np.abs(np.fft.fft(E_Pump))[:len(freqs)]
    E_Ref_Phase = -np.unwrap(np.angle(np.fft.fft(E_Ref)))[:len(freqs)]
    E_Pump_Phase = -np.unwrap(np.angle(np.fft.fft(E_Pump)))[:len(freqs)]

    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi
    R = R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))

    freqs = freqs * 1e12
    mask = (freqs > 0) & (freqs <= 10e12)
    E_Ref_Amp = E_Ref_Amp[mask]
    E_Pump_Amp = E_Pump_Amp[mask]
    E_Ref_Phase = E_Ref_Phase[mask]
    E_Pump_Phase = E_Pump_Phase[mask]
    freqs = freqs[mask]
    R_Amp = R_Amp[mask]
    Phi = Phi[mask]
    R = R[mask]
    return R, E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs



def fit_data(R, freqs, init):
    bounds = ([1e-13, 1e12], [1e-11, 1e14])
    result = least_squares(fitfun, init, bounds=bounds, args=(R, freqs))
    tau_fit, wp_fit = result.x
    sigma_drude_fit = calculate_sigma_drude(tau_fit, wp_fit, freqs)
    eps_fit = calculate_eps(freqs, tau_fit, wp_fit) + sigma_drude_fit / (1j * eps0 * freqs)
    n_fit = np.sqrt(eps_fit)
    return n_fit, sigma_drude_fit, tau_fit, wp_fit, eps_fit


if __name__ == "__main__":
    lockin1_file = "processed_data_temp200/LockIn1.csv"
    lockin2_file = "processed_data_temp200/LockIn2.csv"
    output_folder = "reflection_analysis_new_temp200"
    lockin1_data = pd.read_csv(lockin1_file)
    pump_delays = lockin1_data.columns[1:].tolist()  # Get pump delays from the column names, excluding the first column
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    class Index:
        def __init__(self):
            self.ind = 0

        def next(self, _):
            self.ind += 1
            if self.ind >= len(pump_delays):
                self.ind = 0
            update_plot(callback)

        def prev(self, _):
            self.ind -= 1
            if self.ind < 0:
                self.ind = len(pump_delays) - 1
            update_plot(self)

    def update_plot(self):
        axs[0].cla()
        axs[1].cla()
        pump_delay = pump_delays[callback.ind]
        E_Ref_windowed, E_Pump_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column=pump_delay)
        R, E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, freqs = analyze_reflection(E_Ref_windowed, E_Pump_windowed, time_axis)
        initial_guess = [(1/9.85e+12), 6.8*1e13]
        _, sigma_drude_fit,  tau_fit, wp_fit, eps_fit = fit_data(R, freqs, initial_guess)
        _, r0 = photoinduced_conductivity(sigma_drude_fit, freqs, Z0, d, unpumped_refractive_index_file="refractive_index_data_dec1.csv")
        delta_sigma = calculate_transient_photoconductivity(E_Ref_Amp, E_Pump_Amp, E_Ref_Phase, E_Pump_Phase, r0)
        print(f"Pump delay: {pump_delay} ps")
        print(f"Fitted tau: {tau_fit:.2e} s")
        print(f"Fitted wp: {wp_fit:.2e} rad/s")
        # Append the pump delay and fitted parameters to a list
        if not hasattr(callback, 'results_list'):
            callback.results_list = []

        callback.results_list.append({
            "Pump Delay (ps)": pump_delay,
            "Fitted Tau (s)": f"{tau_fit:.2e}",
            "Fitted wp (rad/s)": f"{wp_fit:.2e}"
        })

        
        axs[0].set_title(f"Drude Conductivity, pump delay= {pump_delay} ps")
        axs[0].plot(time_axis, E_Ref_windowed, label="E_Ref")
        axs[0].plot(time_axis,  E_Pump_windowed,label="E_Pump")
        axs[0].set_xlabel("Time (ps)")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].legend()
        #axs[0].set_xlim(1, 10)
        axs[0].grid(True)

        axs[1].set_title(f"Transient Photoconductivity, pump delay= {pump_delay} ps")
        axs[1].plot(freqs / 1e12, np.imag(delta_sigma), label="Imag(delta_sigma)")
        axs[1].plot(freqs / 1e12, np.real(delta_sigma), label="Real(delta_sigma)")
        axs[1].set_xlabel("Frequency (THz)")
        axs[1].set_ylabel("Transient Photoconductivity (S/m)")
        axs[1].legend()
        #axs[1].set_xlim(1, 10)
        axs[1].grid(True)

        plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.01, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    update_plot(callback)
    plt.show()

    # Save the results list to a single CSV file
    results_df = pd.DataFrame(callback.results_list)
    output_file = os.path.join(output_folder, "fit_results.csv")
    results_df.to_csv(output_file, index=False)
