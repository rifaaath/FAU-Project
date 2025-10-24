import argparse
import shutil
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def organize_glyphs(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    json_path = Path(args.metadata_json)
    excel_path = Path(args.metadata_excel)
    # source_tag = args.source_tag

    # Ensure a clean start 
    if output_dir.exists():
        print(f"Output directory '{output_dir}' already exists. Removing it for a clean run.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metadata once 
    print("Loading metadata...")
    with open(json_path) as f:
        data = json.load(f)
    image_id_to_file = {img["id"]: Path(img["file_name"]).stem.lower().strip() for img in data["images"]}

    df_meta = pd.read_excel(excel_path)
    df_meta["Image Name"] = df_meta["Image Name"].astype(str).str.strip().str.lower()
    file_to_tm = {name: tm for name, tm in zip(df_meta["Image Name"], df_meta["TM with READ item name in ()"])}
    # End Metadata Loading 

    skipped_count = 0
    copy_count = 0

    glyph_files = list(input_dir.glob("*.jpg"))
    for glyph_path in tqdm(glyph_files, desc=f"Organizing glyphs from '{input_dir.name}'"):
        try:
            image_id = int(glyph_path.stem.split("_")[0])

            file_stem = image_id_to_file.get(image_id)
            if not file_stem:
                skipped_count += 1
                continue

            tm_id_raw = file_to_tm.get(file_stem)
            if not tm_id_raw:
                skipped_count += 1
                continue

            tm_id = f"TM_{tm_id_raw}"
            tm_folder = output_dir / tm_id
            tm_folder.mkdir(exist_ok=True)

            # Use the existing filename, as it's already unique
            shutil.copy(glyph_path, tm_folder / glyph_path.name)
            copy_count += 1

        except Exception as e:
            # print(f"Warning: Failed to process {glyph_path.name}. Error: {e}")
            skipped_count += 1
            continue

    print(f"\nOrganization complete.")
    print(f"Total glyphs copied: {copy_count}")
    print(f"Total skipped (no metadata): {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize a flat directory of glyphs into subfolders by writer TM_ID.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    parser.add_argument("--metadata_excel", type=str, required=True)
    args = parser.parse_args()
    organize_glyphs(args)