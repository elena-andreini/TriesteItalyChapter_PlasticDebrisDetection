# 🧠 Dataset Preparation Tool

This script is designed to help with preparing image datasets for machine learning tasks that involve **paired images and masks**. It searches for image-mask pairs, splits them into training, validation, and test sets, and saves the paths into text files for easy loading later.

---

## 📂 Directory Structure

The expected structure of your dataset should look like this:

```
output_batches/
└── patches/
    ├── subfolder1/
    │   ├── image1.tif
    │   ├── image1_cl.tif
    │   └── ...
    └── subfolder2/
        ├── ...
```

- Each subfolder contains `.tif` images.
- Each image should have a corresponding mask file with `_cl.tif` suffix.

---

## 🛠️ Setup

### 1. Clone the repository or copy the script:
Save the script as `prepare_dataset.py`.

### 2. Install required dependencies:

```bash
pip install scikit-learn
```

---

## 🚀 Usage

Run the script from your terminal:

```bash
python prepare_dataset.py --dataset_dir ./output_batches --output_dir ./data
```

### Command-line Arguments

| Argument        | Default            | Description                                               |
|-----------------|--------------------|-----------------------------------------------------------|
| `--dataset_dir` | `./output_batches` | Root directory containing the `patches` subdirectory      |
| `--output_dir`  | `./data`           | Where to save the `splits` (text files with paths)        |

---

## 📄 Output

After running the script, you will get:

```
data/
└── splits/
    ├── train_X.txt
    ├── val_X.txt
    ├── test_X.txt
    ├── train_masks.txt
    ├── val_masks.txt
    └── test_masks.txt
```

Each text file contains **absolute paths** to the images and masks in its respective dataset split.

---

## ✅ Example Output Log

```bash
2025-05-10 14:35:00 - INFO - Created directory: /full/path/to/data
2025-05-10 14:35:00 - INFO - Searching for patches in: /full/path/to/output_batches/patches
2025-05-10 14:35:01 - INFO - Dataset Split - Train: 800, Val: 100, Test: 100
2025-05-10 14:35:01 - INFO - Successfully saved dataset splits.
```

---

## 🔒 License

This project is released under the MIT License.
