import csv
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
# Directory where your cropped kai glyphs are
kai_crop_dir = Path("glyph_crops/kai")

# Paths to the metadata files
json_path = Path("papytwin/HomerCompTraining/HomerCompTraining.json")
excel_path = Path("papytwin/1.CompetitionOverview.xlsx")

# Output file
output_csv = Path("kai_manifest.csv")
# --- END CONFIG ---


# --- Step 1: Load all necessary metadata for mapping ---
print("Loading metadata...")

# Load image_id -> file_stem from JSON
with open(json_path) as f:
    data = json.load(f)
image_id_to_file_stem = {
    img["id"]: Path(img["file_name"]).stem.lower().strip()
    for img in data["images"]
}

# Load file_stem -> TM ID from Excel
df_meta = pd.read_excel(excel_path)
df_meta["Image Name"] = df_meta["Image Name"].astype(str).str.strip().str.lower()
file_stem_to_tm_id = {name: f"TM_{tm}" for name, tm in
                      zip(df_meta["Image Name"], df_meta["TM with READ item name in ()"])}

# --- Step 2: Iterate through the flat directory and create the manifest ---
print(f"Processing glyphs in '{kai_crop_dir}'...")

# List to hold all the valid rows for the CSV
manifest_rows = []
skipped_count = 0

glyph_files = list(kai_crop_dir.glob("*.jpg"))
for glyph_path in tqdm(glyph_files, desc="Creating manifest"):
    try:
        # Extract the original image ID from the glyph's filename
        # e.g., from '3273_30056_cat1__opencv.jpg' -> 3273
        image_id = int(glyph_path.stem.split("_")[0])

        # --- Map back to Writer ID ---
        # 1. Get the original document's file stem (e.g., 'p.oxy.vi.851')
        file_stem = image_id_to_file_stem.get(image_id)
        if not file_stem:
            skipped_count += 1
            continue

        # 2. Get the TM ID from the file stem
        tm_id = file_stem_to_tm_id.get(file_stem)
        if not tm_id:
            skipped_count += 1
            continue

        # Determine the source (opencv/kornia)
        source = "kornia" if "__kornia" in glyph_path.name else "opencv"

        # Add the valid row to our list
        manifest_rows.append({
            "path": glyph_path.as_posix(),
            "tm_id": tm_id,
            "source": source
        })

    except Exception as e:
        # Catch any errors from filenames that don't match the expected format
        # print(f"Warning: Could not process {glyph_path.name}. Error: {e}")
        skipped_count += 1
        continue

# --- Step 3: Write the manifest file ---
print(f"\nFound {len(manifest_rows)} valid glyphs. Skipped {skipped_count} glyphs.")
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "tm_id", "source"])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"✅ Manifest for kai glyphs saved to: {output_csv.resolve()}")