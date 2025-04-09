# Utility Functions & Visualizer Documentation

This markdown file documents key utility functions and the `Visualizer` class used in a satellite image segmentation pipeline.

---

## 🧮 Utility Functions

### `find_unique_in_raw_masks(mask_paths)`
Returns the sorted set of unique class values across a list of raw mask image files.

**Parameters:**
- `mask_paths` (list of str): List of paths to mask images.

**Returns:**
- List of sorted unique class indices.

---

### `apply_uncertainty_class(mask, conf, threshold=1)`
Applies a specific class ID (e.g., 16) to pixels with confidence below a threshold.

**Parameters:**
- `mask` (np.ndarray): The segmentation mask.
- `conf` (np.ndarray): The confidence map.
- `threshold` (int): Pixels below this threshold are set to class 16.

**Returns:**
- Updated `mask` with uncertain areas labeled.

---

### `plot_mask_class_histogram(...)`
Displays a histogram of class frequency in masks using the provided class and color mappings.

**Parameters:**
- `mask_paths` or `dataset`: List of mask file paths or tf.data.Dataset object.
- `class_mapping` (dict): Mapping from class index to label.
- `color_mapping` (dict): Mapping from class index to RGB values.
- `title` (str): Title for the plot.
- `use_merged_labels` (bool): If True, merges class indices using `merge_map`.
- `merge_map` (dict): Class ID remapping for merged categories.

---

### `compute_class_weights(dataset)`
Computes inverse-frequency class weights from a dataset.

**Parameters:**
- `dataset` (tf.data.Dataset): Dataset of image-mask pairs.

**Returns:**
- `np.ndarray`: Weight per class for imbalanced class handling.

---

### `find_unique_classes(dataset)`
Finds all unique class labels present in a dataset.

**Parameters:**
- `dataset`: TensorFlow dataset.

**Returns:**
- Sorted list of unique class IDs.

---

### `check_class_distribution(dataset)`
Counts how frequently each class occurs across the dataset.

**Parameters:**
- `dataset`: TensorFlow dataset.

**Returns:**
- Array of class frequencies.

---

## 🖼️ Class: `Visualizer`

### Purpose:
To visualize the satellite image, segmentation mask, and confidence map for analysis.

### **Initialization Parameters**
- `pipeline` (`DataPipeline`): Instance for loading and processing data.
- `class_mapping` (`dict`): Mapping from class index to human-readable name.
- `color_mapping` (`dict`): Mapping from class index to RGB tuple.
- `confidence_mapping` (`dict`): Mapping from confidence index to description.

---

### `apply_color_mapping(array)`
Applies RGB coloring to a 2D label map using `color_mapping`.

---

### `visualize_sample(image_path, mask_path, conf_path=None, save_path=None)`
Visualizes a sample image, mask, and optional confidence map.

- Shows RGB composite
- Shows colored mask with legend
- Shows confidence with labels

---

### `process_and_visualize_samples(...)`
Processes multiple samples and visualizes them in batches.

**Parameters:**
- `image_file_path`, `mask_file_path`, `conf_file_path`: Paths to lists of image/mask/confidence files.
- `visualizer`: Instance of `Visualizer`.
- `num_images`: Number of samples to visualize.
- `save_dir`: Output directory for saving images.

---

## ✅ Example Use

```python
visualizer = Visualizer(pipeline, class_mapping, color_mapping, confidence_mapping)
process_and_visualize_samples(
    image_file_path="train_X.txt",
    mask_file_path="train_masks.txt",
    conf_file_path="train_confidence.txt",
    visualizer=visualizer,
    num_images=5
)
```

---