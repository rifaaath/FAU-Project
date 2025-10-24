import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil


def is_likely_modern_character(image_path, threshold=25):
    """
    Heuristic to detect modern characters.
    Modern, clean characters have very few unique colors (e.g., black text, white background).
    Glyphs on papyrus have many shades of brown, beige, grey, etc.
    Returns True if the image is likely a modern, clean character.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False  # Can't process, assume it's okay for now

        # A small 10x10 patch has very few unique colors if it's clean
        h, w, _ = img.shape
        patch = img[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]

        # Count the number of unique colors in the small patch
        unique_colors = np.unique(patch.reshape(-1, patch.shape[2]), axis=0)

        return len(unique_colors) < threshold
    except Exception:
        return False


def sanitize_dataset(args):
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    if not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    print(f"Sanitizing glyphs from '{source_dir}'...")
    print(f"Clean glyphs will be copied to '{dest_dir}'.")

    dest_dir.mkdir(parents=True, exist_ok=True)

    removed_count = 0
    copied_count = 0

    writer_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    for writer_folder in tqdm(writer_folders, desc="Processing writers"):
        dest_writer_folder = dest_dir / writer_folder.name
        dest_writer_folder.mkdir(exist_ok=True)

        for glyph_path in writer_folder.glob("*.jpg"):
            if is_likely_modern_character(glyph_path):
                # print(f"Removing likely artifact: {glyph_path}") # Uncomment for debugging
                removed_count += 1
            else:
                # Copy the clean glyph to the new location
                shutil.copy(glyph_path, dest_writer_folder / glyph_path.name)
                copied_count += 1

    print("\nSanitation Complete ")
    print(f"Glyphs copied (kept): {copied_count}")
    print(f"Likely artifacts removed: {removed_count}")
    print(f"A new, sanitized dataset is available at: '{dest_dir.resolve()}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanitize a cropped glyph dataset by removing likely modern artifacts.")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="glyph_crops_yolo_organized_by_tm",  # Point to your organized but contaminated glyphs
        help="The root directory of the organized glyph dataset to clean."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="glyph_crops_yolo_sanitized",  # New output directory
        help="Directory where the sanitized glyph dataset will be saved."
    )
    args = parser.parse_args()
    sanitize_dataset(args)