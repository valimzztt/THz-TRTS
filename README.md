# THz-TRTS Analysis

This repository contains scripts for analyzing THz-TRTS data, including averaging, extracting pump delays, and fitting the dielectric function.

## Workflow

### Step 1: Compute Averages

- **Script**: `average.py`
- **Description**: Computes the average wavefunction for each file in the dataset.
- **Output**: Averaged `.d24` files.

### Step 2: Generate Pump Delays and THz Waveform CSVs

- **Script**: `2D_trts_map.py`
- **Description**: Extracts pump delays and THz waveforms for Lock-In Amplifiers 1 and 2.
- **Output**:
  - `LockIn1.csv`
  - `LockIn2.csv`

### Step 3: Fit Dielectric Function

- **Script**: `2D_scan_analysis.py`
- **Description**: Analyzes the data and fits the dielectric function.
- **Output**: Fitted parameters and reflection coefficients.

## Additional Info

- Pump size: Refer to `KnifeEdge_Dec11.png`.
- Pump energy: 45 μJ.

## Running the Scripts

1. Run `average.py`
   ```bash
   python average.py
   ```
2. Run `2D_trts_map.py`
   ```bash
   python 2D_trts_map.py
   ```
3. Run `2D_scan_analysis.py`
   ```bash
   python 2D_scan_analysis.py
   