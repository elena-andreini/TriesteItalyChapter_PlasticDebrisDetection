
# Sentinel-2 SAFE File Processing Report using ACOLITE

## 📦 Input Overview

- **File**: `S2A_MSIL1C_20190623T101031_N0500_R022_T32TQQ_20230619T042008.SAFE`
- **Satellite**: Sentinel-2A
- **Sensor**: MSI (MultiSpectral Instrument)
- **Processing Tool**: ACOLITE (version 20250402.0)
- **Correction Mode**: DSF (Dark Spectrum Fitting)
- **Parameter File**: `L2W_parameters.txt` (must be placed inside .SAFE folder for L2W to work)

## 🔍 Step-by-Step Processing Summary

### 1. Metadata Import and Geometry Setup
- Input automatically identified as Sentinel-2 Level-1C (L1C).
- Geometry metadata loaded: solar/viewing angles, per-pixel projection, etc.
- Purpose: Base input for corrections

### 2. TOA Reflectance (rhot_)
- Top-Of-Atmosphere reflectance computed for 13 bands.
- Output file: `*_L1R.nc`
- Purpose: TOA reflectance post-decoding

### 3. Atmospheric Correction (rhos_)
- Applied DSF algorithm.
- Surface reflectance computed for 11 bands.
- Output file: `*_L2R.nc`
- Purpose: Surface reflectance after atmospheric correction and use for **NDVI, FDI**

### 4. Water Processing (Rrs_)
- User-specified bands: `Rrs_443`, `Rrs_492`, `Rrs_560`, `Rrs_665`
- Output file: `*_L2W.nc`
- Purpose: Water-specific correction for aquatic studies 

### 5. Ancillary Data
- MERRA2 Meteorological Data: ✅
- Ozone & NCEP: ❌ → Default values used:
  - `uoz = 0.30`, `uwv = 1.50`, `pressure = 1013.25`

## 📁 Output Files

| Type         | File Example                                           | Notes                                 |
|--------------|--------------------------------------------------------|----------------------------------------|
| NetCDF       | `*_L1R.nc`, `*_L2R.nc`, `*_L2W.nc`                     | rhot, rhos, Rrs and indices            |
| PNG Visuals  | `*_rgb_rhot.png`, `*_rgb_rhos.png`                    | TOA and corrected RGB composites       |
| GeoTIFFs     | `rhos_665.tif`, `Rrs_560.tif`, etc.                   | Exported manually with geolocation     |
| Indices      | `NDVI.tif`, `FDI.tif`, `TSS.tif`, etc.                | Computed from corrected bands          |
| Logs/Settings| `acolite_run_*.txt`, `*.log`                          | Full processing metadata               |

## 📈 Band Overview

| Band | Wavelength (nm) | Name         | Type | Usage                          |
|------|------------------|--------------|------|--------------------------------|
| B1   | 443              | rhot_443     | TOA  | Coastal/aerosol                |
| B2   | 492              | rhot_492     | TOA  | Blue                           |
| B3   | 560              | rhot_560     | TOA  | Green                          |
| B4   | 665              | rhot_665     | TOA  | Red                            |
| B5   | 704              | rhot_704     | TOA  | Vegetation Red-edge            |
| B6   | 740              | rhot_740     | TOA  | Red-edge                       |
| B7   | 783              | rhot_783     | TOA  | Red-edge                       |
| B8   | 833              | rhot_833     | TOA  | NIR                            |
| B8A  | 865              | rhot_865     | TOA  | Narrow NIR                     |
| B9   | 945              | rhot_945     | TOA  | Water vapor                    |
| B10  | 1373             | rhot_1373    | TOA  | Cirrus detection               |
| B11  | 1614             | rhot_1614    | TOA  | SWIR1                          |
| B12  | 2202             | rhot_2202    | TOA  | SWIR2                          |

**Source** : [**sentinel-hub.com**](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/)

**Corrected rhos_ bands**: All except B9 & B10  
**Rrs_ bands generated**: `Rrs_443,Rrs_492,Rrs_560,Rrs_665,Rrs_833,Rrs_1614,Rrs_2202`

## 🧮 Indices Calculated

- **NDVI** = (rhos_833 - rhos_665) / (rhos_833 + rhos_665)
- **FDI**  = rhos_783 - rhos_865 + (rhos_865 - rhos_740)
- **TSS**  = A * Rrs_665 / (1 - Rrs_665 / C)   --> where A and C are empirical coefficients based on calibration.
- **CDOM** = Proxy via Rrs_443 / Rrs_490

## 🗺️ Spatial Characteristics

- **CRS**: EPSG:32632 (UTM Zone 32N)
- **Resolution**: 10m/pixel
- **Dimensions**: 10980 × 10980 (≈ 120 km²)
- **GeoTIFF Size**: ~482MB per band

## ❓ Why Some Bands Were Skipped?
Bands B9 (945nm) and B10 (1373nm) were excluded from surface correction:
- Very low atmospheric transmittance: `tgas < 0.7`
- Primarily used for water vapor & cirrus, not surface reflectance.

## 📊 Summary of Output Layers

| Category             | Count | Format(s)                   |
|----------------------|-------|-----------------------------|
| TOA Bands (rhot_*)   | 13    | NetCDF, GeoTIFF             |
| Surface Bands (rhos_)| 11    | NetCDF, GeoTIFF             |
| Water Bands (Rrs_*)  | 4     | NetCDF, GeoTIFF             |
| Indices              | 3     | GeoTIFF (NDVI, FDI, TSS)    |
| RGB Composites       | 2     | PNG + GeoTIFF               |
| Settings & Logs      | 4     | TXT/LOG                     |

## 🎯 Recommendations

- Use `rhos_*` for terrestrial applications.
- Use `Rrs_*` for aquatic/marine studies (plastic debris, CDOM, TSS).
- `rhot_*` may be used for quick TOA-based indices (like NDVI) but not recommended for precise analysis.

## 📚 References

- [ACOLITE GitHub](https://github.com/acolite/acolite/releases)
- [Copernicus Sentinel Hub](https://dataspace.copernicus.eu/analyse/apis/sentinel-hub)
- [ACOLITE Documentation](https://www.sciencedirect.com/science/article/pii/S0034425715000577?via%3Dihub)

---

Prepared using ACOLITE & custom Python workflows (xarray, rioxarray, matplotlib).

## ✅ Recommendations

- Ensure L2W_parameters.txt exists inside the .SAFE folder before running ACOLITE

- Always specify l2w_parameters=Rrs_443 Rrs_490 Rrs_560 Rrs_665 properly (space-separated) in the GUI or .txt

- Use corrected Rrs_* for all water-related indices or machine learning plastic detection

- Validate your .nc file contents using xarray to ensure expected bands are generated

## 🔗 Useful Links

- [ACOLITE GitHub](https://github.com/acolite/acolite)
- [Evaluation of eight band](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13851&id=529038)

- [Copernicus Sentinel-2 Dataspace
](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html#sentinel-2-level-1c-top-of-atmosphere-toa)
-[turbid coastal and estuarine waters with VIIRS I (375 m) and M (750 m) bands](https://www.tandfonline.com/doi/full/10.1080/01431161.2024.2407559#d1e330)
