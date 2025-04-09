# DataPipeline Class Documentation

This module implements a comprehensive image preprocessing pipeline tailored for satellite imagery used in semantic segmentation tasks. The core functionality is encapsulated in the `DataPipeline` class.

---

## 📦 Class: `DataPipeline`

A preprocessing class for satellite images supporting augmentation, normalization, band manipulation, and label mapping for segmentation tasks.

### **Initialization Parameters**

- **`img_size`** (`tuple`): Target size (height, width) of the image after resizing.
- **`num_bands`** (`int`): Number of channels (e.g., spectral bands + indices).
- **`num_classes`** (`int`): Total number of segmentation classes.
- **`rare_classes`** (`list[int]`): Optional. List of rare class indices.
- **`normalization_method`** (`str`): Normalization method: `'percentile'`, `'robust'`, or `'minmax'`.
- **`use_merged_labels`** (`bool`): Whether to use a simplified class mapping.
- **`merge_map`** (`dict`): Optional. Mapping of original class labels to merged labels.

---

## 🔧 Methods

### `normalize_band(band)`
Normalize a single band using the configured method.

### `load_image(image_path)`
Loads a multi-band image from a GeoTIFF file and computes additional indices like NDVI, NDWI, etc.

### `load_mask(mask_path)`
Loads the mask and optionally maps it using `merge_map`.

### `load_confidence(conf_path)`
Loads confidence maps for auxiliary use (e.g., filtering or quality control).

### `has_rare_class(mask)`
Checks if a mask contains any rare classes.

### `apply_augmentation(image, mask)`
Applies random augmentations using the Albumentations library.

### `preprocess_pair(image_path, mask_path, augment=False, augment_rare_only=False)`
Full preprocessing pipeline for a single image-mask pair: normalization, augmentation (optional), and one-hot encoding of the mask.

### `get_tf_dataset(image_paths, mask_paths, batch_size, augment=False, augment_rare_only=False)`
Converts file paths into a TensorFlow dataset object. Handles batching, shuffling, and data loading.

### `summarize_dataset(dataset)`
Prints a summary of the class distributions within the given dataset.

---

## 🗂 Supporting Utilities

### `load_paths(file_path)`
Loads paths from a file, one per line.

---

## 🎨 Class and Color Mappings

Supports full and merged class mappings via dictionaries:
- `class_mapping`, `color_mapping`
- `class_mapping_merged`, `color_mapping_merged`
- `merge_map` — for reducing number of classes

---

## 🧪 Example Usage

```python
pipeline = DataPipeline(
    img_size=(256, 256),
    num_bands=9,
    num_classes=8,
    rare_classes=[1, 2, 3, 4, 5, 6, 7],
    normalization_method='minmax',
    use_merged_labels=True,
    merge_map=merge_map
)

train_dataset = pipeline.get_tf_dataset(train_imgs, train_masks, batch_size=32, augment=True, augment_rare_only=True)
pipeline.summarize_dataset(train_dataset)
```

---

## 🧾 Notes

- Mask one-hot encoding uses `np.eye(num_classes)[mask]`.
- NDVI, NDWI, FDI, and PFDI are used as spectral indices.
- Augmentation pipeline includes flips, rotations, brightness, contrast, and noise.
- The pipeline supports both full and merged class structures.

---

## ✅ Validation

A single batch can be previewed using:

```python
for img, mask in train_dataset.take(1):
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)
```