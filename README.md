# Analysis Explanation: Reflection_2D Function

This files explains the functionality of the following repository concerning the analysis of THz-TRTS data. 

## Key Steps

1. **Initialization**:
    - If the scans for one pump delay were taken separately, then first run average.py; this script will collect all averages and save them inside a new folder that contains the average for each pump delay. 
    - Once this is done, run `2D_initialization.py`, which processes files in the data folder, extract pump probe delays, and saves 2D CSV files for both lock-in amplifiers.
    
2. **Data Import and Baseline Correction**:
    - Import pump times and delays.
    - Read electric field data (`E1` and `E2`) from files.
    - Remove baseline from the data.

3. **Time and Frequency Domain Conversion**:
    - Compute the time and frequency domain representations of the reflected and pumped electric fields.
    - Calculate the amplitude and phase of the electric fields using FFT.

4. **Reflection Coefficient Calculation**:
    - Compute the reflection coefficient `R` using the amplitude and phase of the electric fields.

5. **Material Property Fitting**:
    - Define the fitting function `fitfun` to model the material properties.
    - Use `lsqnonlin` to fit the model to the data and extract parameters like relaxation time (`tau`) and plasma frequency (`wp`).

6. **Drude and Lorentz Model Calculations**:
    - Calculate the Drude and Lorentz components of the dielectric function.
    - Compute the complex refractive index and reflection coefficient.

7. **Interpolation and Final Calculations**:
    - Interpolate the data for finer frequency resolution.
    - Calculate the conductivity, dielectric function, loss, and screening properties.

