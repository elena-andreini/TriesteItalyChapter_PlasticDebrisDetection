import os
import argparse
import urllib.request
import zipfile

def download_and_extract(dataset_path, url):
    """Downloads and extracts the MARIDA dataset."""
    zip_path = os.path.join(dataset_path, "MARIDA.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print("Extraction complete.")

def load_data(source):
    """Loads and downloads the MARIDA dataset from Google Drive (Colab) or a local directory."""
    dataset_path = ""
    dataset_url = "https://zenodo.org/records/5151941/files/MARIDA.zip"
    
    if source == "colab":
        from google.colab import drive
        drive.mount('/content/drive')
        dataset_path = "/content/drive/MyDrive/MARIDA"
    elif source == "local":
        dataset_path = os.path.join(os.getcwd(), "MARIDA")
    else:
        raise ValueError("Invalid source. Use 'colab' or 'local'.")
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    download_and_extract(dataset_path, dataset_url)
    
    print(f"Dataset path: {dataset_path}")
    print("Files:", os.listdir(dataset_path))
    return dataset_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, choices=["colab", "local"], required=True, help="Choose dataset source")
    args = parser.parse_args()
    
    load_data(args.source)
