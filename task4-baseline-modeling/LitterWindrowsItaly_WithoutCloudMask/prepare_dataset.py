import os
import glob
import argparse
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_directory(path):
    """
    Creates a directory if it doesn't already exist.

    Args:
        path (str): Path to the directory.
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Created directory: {os.path.abspath(path)}")

def get_image_and_mask_paths(dataset_path):
    """
    Scans the dataset directory for image and corresponding mask file paths.

    Args:
        dataset_path (str): Root directory containing patch subdirectories.

    Returns:
        tuple: Lists of image paths and corresponding mask paths.
    """
    patches_dir = os.path.join(dataset_path, "patches")
    logger.info(f"Searching for patches in: {os.path.abspath(patches_dir)}")

    image_paths = []
    mask_paths = []

    for subfolder in os.listdir(patches_dir):
        subfolder_path = os.path.join(patches_dir, subfolder)
        if os.path.isdir(subfolder_path):
            images = sorted(glob.glob(os.path.join(subfolder_path, "*.tif")))
            for img_path in images:
                if "_cl.tif" in img_path:
                    continue
                mask_path = img_path.replace(".tif", "_cl.tif")
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

    return image_paths, mask_paths

def split_and_save_data(image_paths, mask_paths, output_dir):
    """
    Splits data into training, validation, and test sets, and saves paths to text files.

    Args:
        image_paths (list): List of image file paths.
        mask_paths (list): List of corresponding mask file paths.
        output_dir (str): Directory where split path files will be saved.
    """
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.5, random_state=42
    )

    logger.info(f"Dataset Split - Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    def save_paths(file_path, paths):
        with open(file_path, "w") as f:
            for path in paths:
                f.write(os.path.abspath(path) + "\n")

    save_paths(os.path.join(splits_dir, "train_X.txt"), train_imgs)
    save_paths(os.path.join(splits_dir, "val_X.txt"), val_imgs)
    save_paths(os.path.join(splits_dir, "test_X.txt"), test_imgs)

    save_paths(os.path.join(splits_dir, "train_masks.txt"), train_masks)
    save_paths(os.path.join(splits_dir, "val_masks.txt"), val_masks)
    save_paths(os.path.join(splits_dir, "test_masks.txt"), test_masks)

    logger.info("Successfully saved dataset splits.")

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare dataset by splitting image and mask paths.")
    parser.add_argument("--dataset_dir", type=str, default="./output_batches", help="Path to dataset root (with patches folder)")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to store split file lists")

    return parser.parse_args()

def main():
    """
    Main function to process the dataset and split image/mask paths.
    """
    args = parse_args()
    create_directory(args.output_dir)
    image_paths, mask_paths = get_image_and_mask_paths(args.dataset_dir)
    split_and_save_data(image_paths, mask_paths, args.output_dir)

if __name__ == "__main__":
    main()
