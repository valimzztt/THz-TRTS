
import numpy as np
import scipy.constants as const

Z0 = 376.73  # Impedance of free space (Ohms)

def calculate_carrier_density(wp):
    """
    Calculates the carrier density from the fitted plasma frequency

    Parameters:
    - wp: plasma frequency (rad/s)

    Returns:
    - eps: Complex dielectric function array.
    """
    factor = 0.6 # Set the effective mass to be 0.6
    effective_mass = const.m_e*factor
    # Carrier Density
    N = (wp**2*effective_mass*const.epsilon_0)/((const.e)**2*Z0)
    return N

wp = 6.80e+13 # rad/s
wp_THz = wp/(2*np.pi*1e12)
print("Plasma frequency in THz:", wp_THz)
N = calculate_carrier_density(wp)
N_cm = N/(1e6)
print(f"Carrier density (m^(-3)): {N:.2e}")
print(f"Carrier density (cm^(-3)): {N_cm:.2e}")
