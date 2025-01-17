import numpy as np
d = 7.56e-7 # Penetration depth (m) (1micrometer)
Z0 = 376.73  # Impedance of free space (Ohms)

def calculate_reflection_coefficient(dE, E_Ref):
    """
    Calculates the complex reflection coefficient from dE and E_Ref.
    """
    r_tilde = dE / E_Ref + 1
    return r_tilde

def calculate_reflectivity_fit(n, sigma_drude):
    r0 = (1 - n) / (1 + n)
    Re = -(1+r0)/r0 * ( Z0*d*sigma_drude ) / ( 1 + n + Z0*d*sigma_drude ) + 1
    return r0, Re


def calculate_reflectivity(E_Ref, E_Pump, time_axis):
    """
    Calculates reflectivity as a function of frequency and pump delay.
    """
    # Compute the frequency axis
    dt = time_axis[1] - time_axis[0]  # Time step
    freqs = np.fft.fftfreq(len(time_axis), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Keep only positive frequencies
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
    return R,  freqs


def calculate_reflectivity2D(E_Ref, E_Pump, time_axis):
    """
    Calculates reflectivity as a function of frequency and pump delay.
    """
    # Compute the frequency axis
    dt = time_axis[1] - time_axis[0]  # Time step
    freqs = np.fft.fftfreq(len(time_axis), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Keep only positive frequencies

    # Compute reflectivity
    E_Ref_Amp = np.abs(np.fft.fft(E_Ref))[:len(freqs), :]
    E_Pump_Amp = np.abs(np.fft.fft(E_Pump))[:len(freqs), :]
    E_Ref_Phase = -np.unwrap(np.angle(np.fft.fft(E_Ref)))[:len(freqs), :]
    E_Pump_Phase = -np.unwrap(np.angle(np.fft.fft(E_Pump)))[:len(freqs), :]
    R_Amp = E_Pump_Amp / E_Ref_Amp
    Phi = E_Pump_Phase - E_Ref_Phase
    Phi = np.mod(Phi + np.pi, 2 * np.pi) - np.pi
    R = R_Amp * (np.cos(Phi) + 1j * np.sin(Phi))
    return R,  freqs

def calculate_differential_reflectivity(E_Ref, E_Pump, time_axis):
    """
    Calculates differential reflectivity as a function of frequency and pump delay.
    """
    # Compute the frequency axis
    dt = time_axis[1] - time_axis[0]  # Time step
    freqs = np.fft.fftfreq(len(time_axis), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Keep only positive frequencies

    # Fourier transform the fields
    E_Ref_fft = np.fft.fft(E_Ref, axis=0)
    E_Pump_fft = np.fft.fft(E_Pump, axis=0)

    # Compute differential reflectivity
    delta_E = E_Pump_fft[:len(freqs)] - E_Ref_fft[:len(freqs)]
    E_Ref_fft = E_Ref_fft[:len(freqs)]
    dR = delta_E / E_Ref_fft
    freqs = freqs * 1e12
    mask = (freqs > 0e12) & (freqs <= 10e12)
    freqs = freqs[mask]
    dR = dR[mask]
    return dR,  freqs

def calculate_differential_reflectivity2D(E_Ref, E_Pump, time_axis):
    """
    Calculates differential reflectivity as a function of frequency and pump delay.
    """
    # Compute the frequency axis
    dt = time_axis[1] - time_axis[0]  # Time step
    freqs = np.fft.fftfreq(len(time_axis), d=dt)
    freqs = freqs[:len(freqs) // 2]  # Keep only positive frequencies

    # Fourier transform the fields
    E_Ref_fft = np.fft.fft(E_Ref, axis=0)
    E_Pump_fft = np.fft.fft(E_Pump, axis=0)

    # Compute differential reflectivity
    delta_E = E_Pump_fft[:len(freqs), :] - E_Ref_fft[:len(freqs), :]
    E_Ref_fft = E_Ref_fft[:len(freqs), :]
    dR = delta_E / E_Ref_fft

    freqs = freqs * 1e12
    mask = (freqs > 0e12) & (freqs <= 10e12)
    freqs = freqs[mask]
    dR = dR[mask]
    return dR,  freqs

