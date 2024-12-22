# Initialization values
import numpy as np 
tl = np.arange(1, 251)  # Time plotted range
fl = np.arange(1, 91)   # Frequency plotted range
ffl = np.arange(5, 42)  # Frequency fitted range
tpl = np.arange(1, 41)  # Time plotted range

Averages = 3
LockIn_1 = 1
LockIn_2 = 2

# Print to verify initialization
print("Initialization values:")
print(f"Time plotted range (tl): {tl}")
print(f"Averages: {Averages}, LockIn_1: {LockIn_1}, LockIn_2: {LockIn_2}")
