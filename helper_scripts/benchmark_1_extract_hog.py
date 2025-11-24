# benchmark_1_extract_hog.py
import argparse
import cv2
import numpy as np
import pandas as pd
import pickle
from skimage.feature import hog
from tqdm import tqdm
from pathlib import Path

def get_doc_id_from_path(path_str):
    """
    Infers a document ID from a glyph's file path.
    Matches the logic in your `prep_final_dataset.py` script.
    Example: '.../31341_0_cat2.jpg' -> '31341_0'
    """
    parts = Path(path_str).stem.split('_')
    # Assumes format like {image_id}_{glyph_index}_cat{category_id}
    return f"{parts[0]}_{parts[1]}" if len(parts) > 2 else Path(path_str).stem

def extract_hog_features(args):
    """
    Iterates through a manifest of pre-cropped glyphs, extracts HOG features,
    and saves them to a pickle file.
    """
    manifest_path = Path(args.manifest_csv)
    output_path = Path(args.output_pickle)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # --- Parameters from the original HoG.py ---
    resize_shape = (64, 64)
    hog_params = dict(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    # -------------------------------------------

    features, labels, sample_ids = [], [], []

    print(f"Extracting HOG features from {len(df)} glyphs...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        glyph_path = row['path']
        try:
            # Load image in grayscale
            gray_img = cv2.imread(glyph_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue

            binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 4)

            # Skip very small, likely invalid crops
            if binary_img.shape[0] < 8 or binary_img.shape[1] < 8:
                continue

            # Resize and compute HOG
            char_crop_resized = cv2.resize(binary_img, resize_shape)
            feature = hog(char_crop_resized, **hog_params)

            # Append data
            features.append(feature)
            # Use 'category_id' as the class label for the codebook
            labels.append(row['category_id'])
            # The 'sample_id' is the document ID, used for grouping
            sample_ids.append(get_doc_id_from_path(glyph_path))
        except Exception as e:
            print(f"Warning: Failed to process {glyph_path}. Error: {e}")
            continue

    with open(output_path, "wb") as f:
        pickle.dump({
            "features": np.array(features),
            "labels": np.array(labels),
            "sample_ids": np.array(sample_ids)
        }, f)

    print(f"\nSaved {len(features)} HOG features to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HOG features for a manifest of cropped glyphs.")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Path to the manifest CSV file (train or test).")
    parser.add_argument("--output_pickle", type=str, required=True, help="Path to save the output .pkl file.")
    args = parser.parse_args()
    extract_hog_features(args)