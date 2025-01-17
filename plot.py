import matplotlib.pyplot as plt
import numpy as np
from src.utils import *

def plot_data(option, **kwargs):
    """
    Plots different aspects of the analysis based on user selection.

    Parameters:
        option (str): The type of plot to generate. Options include:
            - "raw_data": Plots the raw LockIn signals.
            - "non_windowed": Plots the non-windowed E_Ref and E_Pump signals.
            - "windowed": Plots the windowed E_Ref and E_Pump signals.
            - "pulse_in_time": Plots the pulse in the time domain.
            - "power_spectrum": Plots the power spectrum.
            - "conductivity": Plots the experimental and fitted conductivity.
            - "reflectivity": Plots the experimental and calculated reflectivity.
        kwargs: Additional data required for plotting (e.g., signals, frequencies).

    """
    if option == "raw_data":
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['time_raw'], kwargs['E_Ref_raw'], label="E_Ref Raw Data")
        plt.plot(kwargs['time_raw'], kwargs['dE_raw'], label="dE Raw Data")
        plt.xlabel("Time (ps)")
        plt.ylabel("Signal")
        plt.title("Raw LockIn Signals")
        plt.legend()
        plt.grid(True)
        plt.show()


    elif option == "windowed":
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['time_axis'], kwargs['E_Ref_windowed'], label="E_Ref Windowed")
        plt.plot(kwargs['time_axis'], kwargs['E_Pump_windowed'], label="E_Pump Windowed")
        plt.xlabel("Time (ps)")
        plt.ylabel("Signal")
        plt.title("Windowed Signals")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif option == "Differential Field":
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['time_axis'], kwargs['E_Ref_windowed'], label="E_Ref Windowed")
        plt.plot(kwargs['time_axis'], kwargs['dE_windowed'], label="dE Windowed", color="red")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude")
        plt.title("Pulse in Time Domain")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif option == "power_spectrum":
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['freqs'] , np.abs(kwargs['E_Ref_FFT']), label="E_Ref FFT")
        plt.plot(kwargs['freqs'] , np.abs(kwargs['E_Pump_FFT']), label="E_Pump FFT")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Power Spectrum")
        plt.title("Power Spectrum")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif option == "conductivity":
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['freqs'] / 1e12, np.real(kwargs['conductivity']), label="Real(Experimental Conductivity)", linestyle="dotted")
        plt.plot(kwargs['freqs'] / 1e12, np.imag(kwargs['conductivity']), label="Imag(Experimental Conductivity)", linestyle="dotted")
        plt.plot(kwargs['freqs'] / 1e12, np.real(kwargs['conductivity_total']), label="Real(Fitted Conductivity)")
        plt.plot(kwargs['freqs'] / 1e12, np.imag(kwargs['conductivity_total']), label="Imag(Fitted Conductivity)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Conductivity")
        plt.title("Conductivity Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif option == "reflectivity":
        plt.figure(figsize=(10, 6))
        plt.plot(kwargs['freqs'] / 1e12, np.real(kwargs['R_experimental']), label="Real(R_exp)")
        plt.plot(kwargs['freqs'] / 1e12, np.imag(kwargs['R_experimental']), label="Imag(R_exp)")
        plt.plot(kwargs['freqs'] / 1e12, np.real(kwargs['R_calculated']), label="Real(R_fit)", linestyle="--")
        plt.plot(kwargs['freqs'] / 1e12, np.imag(kwargs['R_calculated']), label="Imag(R_fit)", linestyle="--")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Reflectivity")
        plt.title("Reflectivity Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    lockin1_file = "processed_data_temp200/LockIn1.csv"
    lockin2_file = "processed_data_temp200/LockIn2.csv"
    output_folder = "reflection_analysis_temp200"
    pump_delay = "-0.3"
    plotting = True
    omega_0 = 2 * np.pi * 5e12  # Resonance frequency (rad/s), 5 THz
    E_Ref, E_Pump, dE = read_data(lockin1_file, lockin2_file, time_delay_column=pump_delay)
    E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis = process_data(lockin1_file, lockin2_file, time_delay_column=pump_delay)

    if plotting:
        plot_data("raw_data", time_raw=time_axis, E_Ref_raw=E_Ref, dE_raw=dE)
        plot_data("windowed Time signal", time_axis=time_axis, E_Ref_windowed=E_Ref_windowed, E_Pump_windowed=E_Pump_windowed)
        plot_data("Differential Field", time_axis=time_axis, E_Ref_windowed=E_Ref_windowed, dE_windowed=dE_windowed)
        E_Ref_FFT, E_Pump_FFT, freqs = fft(E_Ref_windowed, E_Pump_windowed, time_axis)
        plot_data("power_spectrum", freqs=freqs, E_Ref_FFT=E_Ref_FFT, E_Pump_FFT=E_Pump_FFT)

        initial_guess = [1.00e-13, 6.8e13]  # Initial guess for tau and wp