# standardize_images_safe.py
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


ORIGINAL_DATASET_ROOT = Path("papytwin/HomerCompTraining/")
CLEAN_DATASET_ROOT = Path("datasets/HomerComp_Cleaned/")

def process_and_copy_image(src_path: Path, dest_root: Path):
    """
    Reads an image, standardizes it to JPG, and saves it to a new location,
    preserving the directory structure.
    Returns a status string: 'ok', 'converted', or 'corrupt'.
    """
    try:
        relative_path = src_path.relative_to(ORIGINAL_DATASET_ROOT)
        dest_path = (dest_root / relative_path).with_suffix('.jpg')

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as img:
            # Check if conversion is needed
            is_png_mismatch = (src_path.suffix.lower() == ".png" and img.format != 'PNG')
            needs_rgb_conversion = img.mode in ('RGBA', 'LA', 'P')  # P is for indexed palette

            # If no changes needed and it's already a JPG, just copy it
            if not is_png_mismatch and not needs_rgb_conversion and src_path.suffix.lower() in ['.jpg', '.jpeg']:
                # For safety, let's still open and re-save to fix potential truncation
                img.save(dest_path, 'jpeg', quality=95)
                return "ok"

            # If conversion is needed, convert to RGB before saving as JPEG
            if needs_rgb_conversion:
                img = img.convert('RGB')

            img.save(dest_path, 'jpeg', quality=95)

            status = "converted"
            if is_png_mismatch:
                print(f"\n[!] Mismatched format: {src_path.name} was a {img.format}. Converted to JPG.")

            return status

    except Exception as e:
        print(f"\n[!!] CRITICAL ERROR: Could not process {src_path.name}.")
        print(f"     Reason: {e}")
        return "corrupt"


if __name__ == "__main__":
    print(f"Scanning original images in: {ORIGINAL_DATASET_ROOT}")
    print(f"Saving standardized images to: {CLEAN_DATASET_ROOT}")

    # Ensure the clean output directory exists
    CLEAN_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    # Find all relevant image files, case-insensitive
    image_paths = list(ORIGINAL_DATASET_ROOT.glob("**/*.[pP][nN][gG]")) + \
                  list(ORIGINAL_DATASET_ROOT.glob("**/*.[jJ][pP][gG]")) + \
                  list(ORIGINAL_DATASET_ROOT.glob("**/*.[jJ][pP][eE][gG]"))

    if not image_paths:
        print("No images found. Please check the ORIGINAL_DATASET_ROOT path.")
        exit()

    results = {"ok": 0, "converted": 0, "corrupt": 0}
    corrupt_source_files = []

    for src_path in tqdm(image_paths, desc="Processing images"):
        status = process_and_copy_image(src_path, CLEAN_DATASET_ROOT)
        results[status] += 1
        if status == "corrupt":
            corrupt_source_files.append(src_path)

    print("\nStandardization Complete ")
    print(f"Images processed successfully (copied/ok): {results['ok']}")
    print(f"Images converted to JPG: {results['converted']}")
    print(f"Corrupt/unreadable source files: {results['corrupt']}")

    if corrupt_source_files:
        print("\nThe following source files are corrupt and were SKIPPED:")
        for f in corrupt_source_files:
            print(f"  - {f}")

    print(f"\nA clean, standardized version of your dataset is now available at '{CLEAN_DATASET_ROOT.resolve()}'")