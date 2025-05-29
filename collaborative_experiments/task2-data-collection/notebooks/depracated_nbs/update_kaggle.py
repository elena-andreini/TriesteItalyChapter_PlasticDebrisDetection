def update_kaggle_dataset_with_zip(
    folder_path,
    title,
    dataset_id,
    description="Zipped Sentinel-2 L1C SAFE folders",
    license_name="CC-BY-SA-4.0"
):
    folder_path = Path(folder_path)
    assert folder_path.exists(), "Folder does not exist!"

    metadata_path = folder_path / "dataset-metadata.json"
    zip_files = [f.name for f in folder_path.glob("*.zip")]

    # If metadata already exists - loads it
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
            existing_paths = {res["path"] for res in existing_metadata.get("resources", [])}
    else:
        existing_metadata = {
            "title": title,
            "id": dataset_id,
            "licenses": [{
                "name": license_name,
                "title": "Creative Commons Attribution Share-Alike 4.0",
                "path": "https://creativecommons.org/licenses/by-sa/4.0/"
            }],
            "resources": []
        }
        existing_paths = set()

    # Add only new zip files to the resources list 
    for zipf in zip_files:
        if zipf not in existing_paths:
            existing_metadata["resources"].append({
                "name": Path(zipf).stem,
                "path": zip,
                "description": f"Zipped .SAFE Sentinel-2: {zipf}",
                "type": "other",
                "format": "zip"
            })

    with open(metadata_path, 'w') as f:
        json.dump(existing_metadata, f, indent=2)

    # Update the dataset on Kaggle
    subprocess.run([
        "kaggle", "datasets", "version",
        "-p", str(folder_path),
        "-m", "Add new zip files",
        "--dir-mode", "zip"
    ], check=True)

update_kaggle_dataset_with_zip(
    folder_path="/Users/sara_mac/Desktop/projects/plastic_detection/Sentinel2PlasticDetectionProject/task2-data-collection/kaggle_dataset/Po_River_July_2019",
    title="Litter Rows Italy - Dataset For Plastic Detection Algorithms",
    dataset_id="sarahajbane/litter-windrows",
)

