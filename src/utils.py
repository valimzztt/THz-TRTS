import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(lockin1_file, lockin2_file, time_delay_column):
    lockin1_data = pd.read_csv(lockin1_file)
    lockin2_data = pd.read_csv(lockin2_file)
    global time_axis
    time_axis = lockin1_data.iloc[:, 0].values
    lockin1_signals = lockin1_data[time_delay_column].values
    lockin2_signals = lockin2_data[time_delay_column].values
    lockin1_signals -= np.mean(lockin1_signals)
    lockin2_signals -= np.mean(lockin2_signals)
    E_Ref = lockin1_signals + lockin2_signals # should be minus for T=200K
    E_Pump = lockin1_signals - lockin2_signals # should be plus for T=200K, viceversa for T=298K
    dE = E_Pump - E_Ref
    return E_Ref, E_Pump, dE,  time_axis

def process_data(lockin1_file, lockin2_file, time_delay_column):
    lockin1_data = pd.read_csv(lockin1_file)
    lockin2_data = pd.read_csv(lockin2_file)
    time_axis = lockin1_data.iloc[:, 0].values
    lockin1_signals = lockin1_data[time_delay_column].values
    lockin2_signals = lockin2_data[time_delay_column].values
    lockin1_signals -= np.mean(lockin1_signals)
    lockin2_signals -= np.mean(lockin2_signals)
    E_Ref = lockin1_signals + lockin2_signals # should be minus for T=200K
    E_Pump = lockin1_signals - lockin2_signals # should be plus for T=200K, viceversa for T=298K

    global processed_E_Ref, processed_E_Pump
    global E_Ref_windowed, E_Pump_windowed, dE_windowed

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
    dE_windowed = E_Pump_windowed-E_Ref_windowed
    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]

    return E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis


def extend_to_power_of_2(E_Ref_windowed, E_Pump_windowed, dE_windowed, time_axis):
    """
    Extends the signals to the nearest power of 2 by zero-padding.

    Parameters:
        E_Ref_windowed (numpy.ndarray): Reference signal array.
        E_Pump_windowed (numpy.ndarray): Pump signal array.
        dE_windowed (numpy.ndarray): Difference signal array.
        time_axis (numpy.ndarray): Time axis array.

    Returns:
        extended_signals (dict): Dictionary containing the extended signals and time axis.
            - 'E_Ref_extended': Extended reference signal.
            - 'E_Pump_extended': Extended pump signal.
            - 'dE_extended': Extended difference signal.
            - 'time_axis_extended': Extended time axis.
    """
    # Determine the current length of the signals
    original_length = len(time_axis)

    # Find the next power of 2 greater than or equal to the original length
    next_power_of_2 = 2 ** int(np.ceil(np.log2(original_length)))

    # Calculate the zero-padding needed
    pad_length = next_power_of_2 - original_length

    # Zero-pad the signals
    E_Ref_extended = np.pad(E_Ref_windowed, (0, pad_length), mode='constant')
    E_Pump_extended = np.pad(E_Pump_windowed, (0, pad_length), mode='constant')
    dE_extended = np.pad(dE_windowed, (0, pad_length), mode='constant')

    # Extend the time axis to match the new length
    dt = time_axis[1] - time_axis[0]  # Time step
    time_axis_extended = np.arange(0, next_power_of_2 * dt, dt)

    # Ensure the time axis length matches the extended signals
    time_axis_extended = time_axis_extended[:next_power_of_2]
    return E_Ref_extended, E_Pump_extended, dE_extended, time_axis_extended


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
    E_Ref_FFT = np.fft.fft(E_Ref)[:len(freqs)]
    E_Pump_FFT = np.fft.fft(E_Pump)[:len(freqs)]

    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi
    R = R_Amp * (np.cos(Phi) +1j * np.sin(Phi))


    freqs = freqs * 1e12
    mask = (freqs > 0e12) & (freqs <= 10e12)
    E_Ref_Amp = E_Ref_Amp[mask]
    E_Pump_Amp = E_Pump_Amp[mask]
    E_Ref_Phase = E_Ref_Phase[mask]
    E_Pump_Phase = E_Pump_Phase[mask]
    freqs = freqs[mask]
    R_Amp = R_Amp[mask]
    Phi = Phi[mask]
    R = R[mask]
    E_Ref_FFT = E_Ref_FFT[mask]
    E_Pump_FFT = E_Pump_FFT[mask]
    return R, E_Ref_FFT, E_Pump_FFT, freqs


def fft(E_Ref_windowed, E_Pump_windowed, time_axis):
    E_Ref = E_Ref_windowed
    E_Pump = E_Pump_windowed
    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]
    E_Pump_FFT = np.abs(np.fft.fft(E_Pump)[:len(freqs)])
    E_Ref_FFT = np.abs(np.fft.fft(E_Ref)[:len(freqs)])
    return E_Ref_FFT, E_Pump_FFT, freqs

def power_spectrum(E_Ref_windowed, E_Pump_windowed, time_axis):
    """
    Computes and plots the power spectrum of the windowed E_Ref and E_Pump signals.

    Parameters:
    - E_Ref_windowed: Windowed reference electric field.
    - E_Pump_windowed: Windowed pump electric field.
    - time_axis: Time axis corresponding to the signals.

    Returns:
    - E_Ref_power: Power spectrum of the reference electric field.
    - E_Pump_power: Power spectrum of the pump electric field.
    - freqs: Frequency axis.
    """
    # Calculate the FFT
    time_sample = time_axis[1] - time_axis[0]  # Sampling interval
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]  # Positive frequencies only

    # Compute FFT and power spectrum
    E_Ref_FFT = np.fft.fft(E_Ref_windowed)
    E_Pump_FFT = np.fft.fft(E_Pump_windowed)

    E_Ref_power = np.abs(E_Ref_FFT[:len(freqs)])**2
    E_Pump_power = np.abs(E_Pump_FFT[:len(freqs)])**2

    return E_Ref_power, E_Pump_power, freqs


