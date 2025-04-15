# Set the project root
project_root = "folder"
if project_root not in sys.path:
    sys.path.append(project_root)
print(project_root)

# Set Kaggle config directory
KAGGLE_CONFIG_DIR = os.path.expanduser(project_root + ".kaggle")
print(KAGGLE_CONFIG_DIR)

# Path to data directory to Po_River
data_dir = project_root + "kaggle_dataset/Po_River_July_2019/"
if data_dir not in sys.path:
    sys.path.append(data_dir)

# Check for kaggle.json credentials
kaggle_json_path = os.path.expanduser(project_root + "/.kaggle/kaggle.json")
if not os.path.exists(kaggle_json_path):
    raise FileNotFoundError(f"The Kaggle API credentials file is missing. Please place your kaggle.json file at {kaggle_json_path}.")

# Function to create or update a Kaggle dataset from a local folder
def create_kaggle_dataset_from_folder(
    folder_path,
    title,
    dataset_id,
    description="Sentinel-2 L1C subset",
    license_name="CC-BY-SA-4.0"
):
    folder_path = Path(folder_path)
    assert folder_path.exists(), "Folder does not exist!"

    metadata_path = folder_path / "dataset-metadata.json"
    image_files = [f.name for f in folder_path.glob("*.tif*")]

    resources = [
        {
            "name": Path(img).stem,
            "path": img,
            "description": f"Image: {img}",
            "type": "image",
            "format": "tiff"
        } for img in image_files
    ]

    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{
            "name": license_name,
            "title": "Creative Commons Attribution Share-Alike 4.0",
            "path": "https://creativecommons.org/licenses/by-sa/4.0/"
        }],
        "resources": resources
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Initialize if necessary
    if not (folder_path / "dataset-metadata.json").exists():
        subprocess.run(["kaggle", "datasets", "init", "-p", str(folder_path)])

    # Create or version the dataset
    if not any((folder_path / f).exists() for f in ["dataset-metadata.json", "dataset-metadata.yml"]):
        print("No metadata found, initializing dataset.")
        subprocess.run(["kaggle", "datasets", "init", "-u", str(folder_path)])

    try:
        subprocess.run([
            "kaggle", "datasets", "create",
            "-p", str(folder_path),
            "--dir-mode", "zip"
        ], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([
            "kaggle", "datasets", "version",
            "-p", str(folder_path),
            "-m", "Update data",
            "--dir-mode", "zip"
        ])

# Example call to the above function
create_kaggle_dataset_from_folder(
    folder_path="/Users/sara_mac/Desktop/projects/plastic_detection/Sentinel2PlasticDetectionProject/task2-data-collection/kaggle_dataset/Po_River_July_2019",
    title="Litter Rows Dataset for ML",
    dataset_id="sarahajbane/litter-windrows"
)
