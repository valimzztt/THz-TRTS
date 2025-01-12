import math
import scipy.constants as const

# Input parameters
wavelength_m = 800e-9  # Laser wavelength in meters
energy_microJ = 24.7  # Laser pulse energy in microJoules
vertical_cm = 0.1698  # Beam vertical radius in cm
horizontal_cm = 0.2167  # Beam horizontal radius in cm
OD = 0.1  # Optical density at laser excitation wavelength
phi = 1  # Photon conversion efficiency (ideal)

# Convert units and calculate
A_pump_cm = math.pi * vertical_cm * horizontal_cm  # Effective area in cm^2
energy_J = energy_microJ * 1e-6  # Convert energy to Joules
fluence_microJ = energy_microJ / A_pump_cm  # Fluence in microJoules/cm^2
fluence_J = fluence_microJ * 1e-6  # Fluence in Joules/cm^2

T_pump = 10 ** (-OD)  # Transmission fraction
R_pump = 0  # Reflectivity (neglected here)
absorption = 10e5
absorption_fraction = (1 - 10 ** (-OD))  # Absorbed fraction
d_microm = 0.9*1e-1 # penetration depth
d_cm = d_microm*1e-4 # penetration depth
absorption = 1/d_cm

# Photon energy and carrier density calculations
energy_per_photon = const.h * const.c / wavelength_m  # Energy per photon in Joules

# Carrier density using absorption fraction and fluence
def CarrDensity(phi, energy_J, absorption, wavelength, area_eff):
    return phi * energy_J * absorption * wavelength / (const.h * const.c * area_eff)

def CarrDensity2(phi, fluence_J, absorption, wavelength, d_cm ):
    return phi * fluence_J * absorption * wavelength / (const.h * const.c*d_cm)

def CarrDensity3(phi,  energy_J, absorption, wavelength, d_cm, area_eff):
    return phi *  energy_J  * absorption * wavelength / (const.h * const.c*d_cm* area_eff)

# Compute carrier density
carrier_density = CarrDensity(phi,energy_J, absorption, wavelength_m, A_pump_cm)
carrier_density_2 = CarrDensity2(phi, fluence_J,absorption_fraction, wavelength_m, d_cm )
carrier_density_3 = CarrDensity3(phi, energy_J,absorption_fraction, wavelength_m, d_cm,  A_pump_cm )
# Results
print(f"Energy per pulse: {energy_J:.3e} J")
print(f"Energy per photon: {energy_per_photon:.3e} J")
print(f"Fluence: {fluence_microJ:.3f} μJ/cm²")
print(f"Carrier density: {carrier_density:.3e} cm⁻³")
print(f"Carrier density: {carrier_density_2:.3e} cm⁻³")
print(f"Carrier density: {carrier_density_3:.3e} cm⁻³")