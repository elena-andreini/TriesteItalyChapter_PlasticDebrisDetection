# ACOLITE GUI SAFE File Atmospheric Correction Process

This report explains the step-by-step atmospheric correction process performed by ACOLITE GUI for Sentinel-2 SAFE files. It includes inputs, process stages, and outputs, using a real processing log as an illustrative example.

---

## Overview

ACOLITE (Atmospheric Correction for OLI 'Lite') is used to process satellite imagery, particularly from Sentinel-2, to generate surface reflectance products. The GUI allows users to visually select input SAFE files and parameter files and initiate processing with minimal manual command line intervention.

---

## Input Parameters

From the log, the following inputs were selected:

- **Parameter file**: `L2W_parameters.txt`
- **Input SAFE file**: Sentinel-2 L1C product (`.SAFE` directory)
- **Output directory**: Target folder for processed data and intermediate files

---

## Step-by-Step Processing Pipeline

### 1. **Initialization and File Identification**
- ACOLITE initializes and logs environment/platform details (macOS, Python version, etc.).
- Identifies the Sentinel-2 `.SAFE` input file and confirms product type.

### 2. **L1R Conversion**
- **Metadata Import**: ACOLITE reads metadata (e.g., `L1C_T32TQQ_A020897_20190623T101029`).
- **Geometry Calculations**: Computes per-pixel solar zenith angle (sza), view zenith angle (vza), and relative azimuth angle (raa).
- **Geolocation**: Writes longitude and latitude per pixel.
- **TOA Reflectance Conversion**: Converts bands to reflectance at the top-of-atmosphere (TOA), stored as `rhot_<band>`.
- Output: `L1R.nc` file created, and `L1R_rgb_rhot.png` preview generated.

### 3. **Ancillary Data Acquisition**
- Downloads meteorological data (e.g., MERRA2 MET files).
- If ozone/NCEP data are unavailable, default values are used:
  - uoz: 0.30
  - uwv: 1.50
  - pressure: 1013.25

### 4. **Atmospheric Correction (DSF Method)**
- Loads LUTs: `ACOLITE-LUT-202110-MOD1` and `MOD2`
- Performs AOT (Aerosol Optical Thickness) estimation per band (443–865 nm).
- Selects best LUT based on RMSD (MOD2 chosen in this example).
- Computes surface reflectance per band (`rhos_<band>`).

#### Log Snippet – AOT Estimation and Surface Reflectance:
```bash
Running AOT estimation for band 3 (rhot_560)
S2A_MSI/B3 ACOLITE-LUT-202110-MOD1 took 0.001s (RevLUT)
S2A_MSI/B3 ACOLITE-LUT-202110-MOD2 took 0.001s (RevLUT)
...
Selected ACOLITE-LUT-202110-MOD2, mean aot = 0.39
Wrote rhot_560 (10980, 10980)
Computing surface reflectance 3 560 0.934
Interpolating tiles
Wrote rhos_560 (10980, 10980)
```

### 5. **File Output – L2R Product**
- Writes corrected reflectance values and generates visual outputs:
  - `L2R.nc` NetCDF file
  - `rgb_rhos.png` and `rgb_rhot.png` previews

### 6. **Water Processing – L2W Product**
- **Masking and Flagging**:
  - Non-water mask from rhot_1614
  - Cirrus mask from rhot_1373
  - TOA limit and negative reflectance masks
- **Rrs Calculation**:
  - Converts surface reflectance to remote sensing reflectance (Rrs) for bands like 443, 492, 560, 665, 833, 1614, 2202

#### Log Snippet – Rrs Calculation:
```bash
Copying Rrs_443, base dataset rhos_443
Writing Rrs_443
Wrote Rrs_443 (10980, 10980)
...
Wrote Rrs_2202 (10980, 10980)
Wrote l2_flags (10980, 10980)
```
- Final output:
  - `L2W.nc` file
  - Rrs preview images (e.g., `Rrs_443.png`, `Rrs_833.png`, etc.)

---

## Example Outputs

Generated files include:
- `.nc` datasets:
  - `L1R.nc`: Top-of-atmosphere reflectance
  - `L2R.nc`: Surface reflectance
  - `L2W.nc`: Water-corrected reflectance (Rrs)
- PNG quicklooks:
  - `*_rgb_rhot.png`, `*_rgb_rhos.png`, `*_Rrs_*.png`

---

## Summary of Band Reflectances

- **TOA Reflectance (rhot)**: 13 bands (443 to 2202 nm)
- **Surface Reflectance (rhos)**: 11 bands
- **Remote Sensing Reflectance (Rrs)**: 7 bands (443, 492, 560, 665, 833, 1614, 2202) and you can Extended by busing her name of Wavelength (nm) of the band `Rrs_<Wavelength(nm)>` =`Rrs_443`

---

## Conclusion

ACOLITE GUI enables comprehensive atmospheric correction of Sentinel-2 imagery via a visual interface. From metadata parsing to multi-stage correction and visualization, the tool simplifies workflows with minimal user input, ensuring robust outputs including reflectance products and quality masks.
