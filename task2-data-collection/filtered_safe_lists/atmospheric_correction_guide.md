
# 🌤️ Atmospheric Correction: Image Format Guidelines

If you plan to perform **atmospheric correction**, choosing the correct image format is essential. Here's a quick guide:

## ❓ What’s the best image format to download?

| **Use Case**                      | **Preferred Format** | **Why?**                                                                 |
|----------------------------------|----------------------|--------------------------------------------------------------------------|
| 📈 Visual analysis or indices (NDVI, etc.) | `TIFF`                | Simple, fast, and ready for processing                                   |
| 🌍 Atmospheric correction using ACOLITE or Sen2Cor | `.SAFE`               | Required by tools—needs full structure including TOA, metadata, sun angles |

---

## 🔁 Difference between `.SAFE` and `TIFF`

| Feature                            | `.SAFE`                          | `TIFF`                             |
|------------------------------------|----------------------------------|------------------------------------|
| Includes metadata (sun angles, calibration)? | ✅ Yes                          | ❌ No                              |
| File structure                     | Folder with multiple band files  | Single file with all bands         |
| Compatible with ACOLITE / Sen2Cor | ✅ Yes                          | ❌ No                              |
| Direct analysis (rasterio, numpy)?| ❌ Needs ESA tools                | ✅ Yes                              |
| Source                             | ESA/CDSE only                    | APIs like SentinelHub              |

---

## 🔍 Where to get `.SAFE` files?

To perform atmospheric correction, download **Sentinel-2 L1C or L2A** data in `.SAFE` format from:

- 🔗 [Copernicus Data Space](https://dataspace.copernicus.eu/browser)
- `sentinelsat` or `copernicus-dataspace-client`

---

## ✅ Recommendations

| **Goal**                             | **Download Method**        | **Response Format**     |
|-------------------------------------|----------------------------|--------------------------|
| Quick analysis (NDVI, RGB, etc.)    | SentinelHub SDK            | `MimeType.TIFF`          |
| Atmospheric correction (ACOLITE, Sen2Cor) | Use `.SAFE` from ESA/CDSE | Full directory structure |
