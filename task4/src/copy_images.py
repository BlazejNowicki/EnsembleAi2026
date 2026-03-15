import os
import shutil
from PIL import Image


def sort_images_by_size(source_dir, default_dir, label_dir):
    """
    Copies PNG images from a source directory to different directories
    based on their resolution.
    """
    # Create the destination directories if they do not exist
    os.makedirs(default_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Verify the source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: The source directory '{source_dir}' was not found.")
        return

    processed_count = 0

    # Iterate through all files in the source directory
    for filename in os.listdir(source_dir):
        # Process only PNG files (case-insensitive)
        if filename.lower().endswith(".png"):
            file_path = os.path.join(source_dir, filename)

            try:
                # Open the image to check dimensions
                with Image.open(file_path) as img:
                    width, height = img.size

                # Determine the target directory
                if width == 3300 and height == 2550:
                    target_dir = default_dir
                else:
                    target_dir = label_dir

                # Copy the file (copy2 preserves file metadata like creation date)
                shutil.copy2(file_path, target_dir)
                processed_count += 1

                # Optional: Print progress for every 50 images
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} images...")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nDone! Successfully processed {processed_count} PNG images.")


# --- Configuration ---
# Replace these strings with the actual paths on your computer
SOURCE_DIRECTORY = "ecg/test"
DEFAULT_DIRECTORY = "ecg/default"
LABEL_DIRECTORY = "ecg/label"

if __name__ == "__main__":
    sort_images_by_size(SOURCE_DIRECTORY, DEFAULT_DIRECTORY, LABEL_DIRECTORY)
