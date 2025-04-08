import os
import kagglehub
import zipfile
import json
import glob
from sklearn.model_selection import train_test_split

# Constants
BASE_DIR = "../data/"

def download_and_extract_dataset():
    print("Downloading MARIDA dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("anangfath/marida-marine-debrish-dataset")
    print(f"Path to dataset files: {dataset_path}")
    
    if os.path.exists(f"{BASE_DIR}MARIDA.zip"):
        print("Extracting MARIDA dataset...")
        with zipfile.ZipFile(f"{BASE_DIR}MARIDA.zip", 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print("Dataset extracted!")
    return dataset_path

# Create directory for storing processed files
def create_directory(path):
    # PROCESSED_DIR = os.path.join(BASE_DIR)
    if not os.path.exists(path):
        os.makedirs(path)

# Function to get image and mask paths
def get_image_and_mask_paths(dataset_path):
    # Print the absolute path for debugging
    patches_dir=f"{dataset_path}/patches/"
    print(f"Looking for patches in directory: {os.path.abspath(patches_dir)}")
    image_paths = []
    mask_paths = []
    confidence_paths = [] 

    for subfolder in os.listdir(patches_dir):
        subfolder_path = os.path.join(patches_dir, subfolder)
        if os.path.isdir(subfolder_path):
            images = sorted(glob.glob(os.path.join(subfolder_path, "*.tif")))
            for img_path in images:
                if "_cl.tif" in img_path or "_conf.tif" in img_path:
                    continue
                
                mask_path = img_path.replace(".tif", "_cl.tif")
                cof_path = img_path.replace(".tif", "_conf.tif")  # Confidence mask path

                if os.path.exists(mask_path) and os.path.exists(cof_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    confidence_paths.append(cof_path)  
    
    return image_paths, mask_paths, confidence_paths

# Function to split and save data into train, validation, and test sets
def split_and_save_data(image_paths, mask_paths,confidence_paths):
    train_imgs, temp_imgs, train_masks, temp_masks, train_confidence, temp_confidence = train_test_split(
            image_paths, mask_paths, confidence_paths, test_size=0.2, random_state=42
        )
    val_imgs, test_imgs, val_masks, test_masks, val_confidence, test_confidence = train_test_split(
            temp_imgs, temp_masks, temp_confidence, test_size=0.5, random_state=42
        )
    print(f"📊 Dataset Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    # Save paths to text files
    splits_dir = f"{BASE_DIR}splits/"
    os.makedirs(splits_dir, exist_ok=True)
    
    def save_paths(file_path, paths):
        with open(file_path, "w") as f:
            for path in paths:
                f.write(os.path.abspath(path) + "\n")

    
    save_paths(f"{splits_dir}train_X.txt", train_imgs)
    save_paths(f"{splits_dir}val_X.txt", val_imgs)
    save_paths(f"{splits_dir}test_X.txt", test_imgs)
    
    save_paths(f"{splits_dir}train_masks.txt", train_masks)
    save_paths(f"{splits_dir}val_masks.txt", val_masks)
    save_paths(f"{splits_dir}test_masks.txt", test_masks)
    
    save_paths(f"{splits_dir}train_confidence.txt", train_confidence)
    save_paths(f"{splits_dir}val_confidence.txt", val_confidence)
    save_paths(f"{splits_dir}test_confidence.txt", test_confidence)

    print("✅ Successfully split data into train, validation, and test sets with confidence masks!")

# Load the label mapping from a JSON file    
def load_labels(file_path):
    with open(file_path, "r") as f:
        label_mapping = json.load(f)
    return label_mapping


def main():
    dataset_path = download_and_extract_dataset()
    create_directory(BASE_DIR)
    image_paths, mask_paths, confidence_paths = get_image_and_mask_paths(dataset_path)
    split_and_save_data(image_paths, mask_paths, confidence_paths)

    label_mapping = load_labels(f"{dataset_path}/labels_mapping.txt")
    print("Labels loaded successfully.")

if __name__ == "__main__":
    main()