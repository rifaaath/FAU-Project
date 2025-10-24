# A simplified version for datasets without HomerComp metadata (for grk here)
import json, os, cv2, torch, kornia as K, argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def crop_generic(args):
    images_dir = Path(args.images_dir)
    predictions_path = Path(args.predictions_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(predictions_path) as f:
        predictions_by_filename = json.load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for filename, preds in tqdm(predictions_by_filename.items(), desc="Cropping glyphs"):
        if not preds: continue

        image_path = images_dir / filename
        if not image_path.exists():
            print(f"Warning: Image file not found: {image_path}")
            continue

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None: continue
        h_img, w_img = img_bgr.shape[:2]

        # We can create a manifest on the fly
        for i, pred in enumerate(preds):
            try:
                x, y, w, h = map(int, pred["bbox"])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                if x2 <= x1 or y2 <= y1: continue

                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0: continue

                # Create a unique, informative filename for the crop
                # e.g., P_1_gly ph_1_cat_5.jpg
                crop_basename = Path(filename).stem
                crop_path = output_dir / f"{crop_basename}_glyph_{i}_cat{pred['category_id']}.jpg"
                cv2.imwrite(str(crop_path), crop)

            except Exception as e:
                print(f"Error cropping from {filename}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop glyphs for a generic dataset.")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    crop_generic(args)