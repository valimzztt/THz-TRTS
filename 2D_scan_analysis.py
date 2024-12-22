import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

def analyze_reflection(lockin1_file, lockin2_file, output_folder):
    """
    Analyze the reflection data using Lock-In Amplifier CSV files.

    Parameters:
    - lockin1_file (str): Path to the LockIn1 CSV file.
    - lockin2_file (str): Path to the LockIn2 CSV file.
    - output_folder (str): Path to the folder where results will be saved.
    """
    # Load data from CSV files
    lockin1_data = pd.read_csv(lockin1_file)
    lockin2_data = pd.read_csv(lockin2_file)

    # Extract time axis and pump delays
    time_axis = lockin1_data.iloc[:, 0].values  # First column is time
    pump_delays = lockin1_data.columns[1:].astype(float).values

    # Extract signal data
    lockin1_signals = lockin1_data.iloc[:, 1:].values
    lockin2_signals = lockin2_data.iloc[:, 1:].values

    # Subtract baselines
    lockin1_signals -= np.mean(lockin1_signals, axis=0)
    lockin2_signals -= np.mean(lockin2_signals, axis=0)

    # Compute reflection-related quantities
    E_Ref = lockin1_signals - lockin2_signals
    E_Pump = lockin1_signals + lockin2_signals
    dE = 2 * lockin2_signals

    time_sample = time_axis[1] - time_axis[0]
    freqs = np.fft.fftfreq(len(time_axis), d=time_sample)
    freqs = freqs[:len(freqs) // 2]  # Positive frequencies only

    E_Ref_Amp = np.abs(np.fft.fft(E_Ref, axis=0))[:len(freqs), :]
    E_Pump_Amp = np.abs(np.fft.fft(E_Pump, axis=0))[:len(freqs), :]
    E_Ref_Phase = -np.unwrap(np.angle(np.fft.fft(E_Ref, axis=0)))[:len(freqs), :]
    E_Pump_Phase = -np.unwrap(np.angle(np.fft.fft(E_Pump, axis=0)))[:len(freqs), :]

    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi

    R = R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))

if __name__ == "__main__":
    lockin1_file = "processed_data/LockIn1.csv"
    lockin2_file = "processed_data/LockIn2.csv"
    output_folder = "reflection_analysis"

    os.makedirs(output_folder, exist_ok=True)
    analyze_reflection(lockin1_file, lockin2_file, output_folder)
