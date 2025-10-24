import csv
import argparse
from pathlib import Path
import pandas as pd
import re


def create_grk_manifest(args):
    glyph_dir = Path(args.glyph_dir)
    gt_csv_path = Path(args.gt_csv)
    output_csv = Path(args.output_csv)

    print(f"Loading ground truth from: {gt_csv_path}")
    try:
        df_gt = pd.read_csv(gt_csv_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    filename_to_writer = {}

    image_name_col = [col for col in df_gt.columns if 'GRK-papyrus' in col][0]
    print(f"Using column '{image_name_col}' for image and writer info.")

    for _, row in df_gt.iterrows():
        full_name = row[image_name_col]
        if pd.isna(full_name): continue
        image_filename = f"{full_name.replace(' ', '_')}.jpg"
        writer_match = re.match(r'([a-zA-Z]+)', full_name)
        if writer_match:
            writer_id = writer_match.group(1)
            filename_to_writer[image_filename] = writer_id

    if not filename_to_writer:
        print("Error: Could not build a filename-to-writer map.")
        return
    print(f"Successfully created map for {len(filename_to_writer)} images.")

    valid_filenames = set()
    if args.image_list_dir:
        print(f"Filtering manifest to only include images from: {args.image_list_dir}")
        image_list_path = Path(args.image_list_dir)
        for img_path in image_list_path.glob("*.jpg"):
            valid_filenames.add(img_path.name)
        print(f"Found {len(valid_filenames)} images in the target directory.")

    print(f"Scanning glyph directory: {glyph_dir}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['path', 'tm_id', 'source', 'image_filename'])

        found_glyphs = 0
        for glyph_path in glyph_dir.glob("*.jpg"):
            try:
                original_filename = "_".join(glyph_path.stem.split("_")[:-3]) + ".jpg"
                writer_id = filename_to_writer.get(original_filename)

                if writer_id:

                    if args.image_list_dir and original_filename not in valid_filenames:
                        continue

                    writer_csv.writerow([glyph_path.as_posix(), writer_id, 'yolo', original_filename])

                    found_glyphs += 1
            except Exception:
                continue

    print(f"\nManifest created at '{output_csv}' with {found_glyphs} glyph entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a manifest for the GRK Papyri dataset.")
    parser.add_argument("--glyph_dir", required=True)
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--image_list_dir", type=str, default=None,
                        help="Optional: Directory of images to restrict the manifest to.")
    args = parser.parse_args()
    create_grk_manifest(args)