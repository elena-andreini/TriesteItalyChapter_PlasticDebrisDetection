import os
import shutil
import json
import subprocess
from pathlib import Path


project_root = "/content/Sagar_TriesteItalyChapter_PlasticDebrisDetection/task2-data-collection"
dataset_main = Path(project_root) / "dataset_main"
kaggle_json_path = Path(project_root) / ".kaggle/kaggle.json"
kaggle_config_dir = kaggle_json_path.parent
# Initial Upload 
# aoi_folders = [
#     Path(project_root) / "data/SAFE/PO_0319",
#     Path(project_root) / "data/SAFE/CA_0719"
# ]
aoi_folders = [
    Path(project_root) / "data/SAFE/CO_0719",
]
dataset_title = "Litter Rows Dataset2"
dataset_id = "sagarpatel31/sagar-litter-windrows"
license_name = "CC-BY-SA-4.0"


os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_config_dir)
os.chmod(kaggle_json_path, 0o600)

dataset_main.mkdir(parents=True, exist_ok=True)

# === COPY AOI FOLDERS INTO dataset_main ===
for src_folder in aoi_folders:
    aoi_name = src_folder.name
    dest_folder = dataset_main / aoi_name
    dest_folder.mkdir(parents=True, exist_ok=True)

    for zip_file in src_folder.glob("*.zip"):
        dest_file = dest_folder / zip_file.name
        if not dest_file.exists():
            shutil.copy2(zip_file, dest_file)
            print(f"Copied: {zip_file.name} → {aoi_name}")
        else:
            print(f"Skipped (already exists): {zip_file.name} in {aoi_name}")

# === CREATE OR MODIFY METADATA FILE ===
metadata_path = dataset_main / "dataset-metadata.json"
if not metadata_path.exists():
    subprocess.run(["kaggle", "datasets", "init", "-p", str(dataset_main)], check=True)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

metadata["title"] = dataset_title
metadata["id"] = dataset_id
metadata["licenses"] = [{
    "name": license_name,
    "title": "Creative Commons Attribution Share-Alike 4.0",
    "path": "https://creativecommons.org/licenses/by-sa/4.0/"
}]

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

# # === CREATE OR VERSION KAGGLE DATASET ===
# def upload_or_version_kaggle_dataset(folder_path, message="Updated with AOI zip folders"):
#     try:
#         result = subprocess.run([
#             "kaggle", "datasets", "create",
#             "-p", str(folder_path),
#             "--dir-mode", "zip"
#         ], capture_output=True, text=True, check=True)
#         print("Dataset created successfully!")
#         print(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print("Dataset already exists. Creating a new version...")
#         print(e.stderr)
#         version_result = subprocess.run([
#             "kaggle", "datasets", "version",
#             "-p", str(folder_path),
#             "-m", message,
#             "--dir-mode", "zip"
#         ], capture_output=True, text=True)
#         print(version_result.stdout)
#         print(version_result.stderr)

# # === RUN ===
# upload_or_version_kaggle_dataset(dataset_main)

version_message = "Added CO_0719 zip folder"

def version_kaggle_dataset(folder_path, message="Updated dataset"):
    result = subprocess.run([
        "kaggle", "datasets", "version",
        "-p", str(folder_path),
        "-m", message,
        "--dir-mode", "zip"
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Kaggle versioning failed:")
        print(result.stderr)

# === RUN ===
version_kaggle_dataset(dataset_main, version_message)
