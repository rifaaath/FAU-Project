import csv
from pathlib import Path
from tqdm import tqdm

# --- Config ---
# Path to your final, organized glyph directory
merged_dir = Path("glyph_crops_merged_by_tm")
output_csv = Path("glyph_manifest_full.csv")  # New output filename
# --- End Config ---

print(f"Scanning '{merged_dir}' to create a full manifest...")
manifest_rows = []

# Iterate through all writer subfolders
for tm_folder in tqdm(merged_dir.iterdir(), desc="Processing writers"):
    if not tm_folder.is_dir():
        continue

    tm_id = tm_folder.name

    # Iterate through all glyphs in the writer's folder
    for glyph_file in tm_folder.glob("*.jpg"):
        try:
            path = glyph_file.as_posix()
            source = "kornia" if "__kornia" in glyph_file.name else "opencv"

            # --- ✅ NEW LOGIC: Extract category_id from the filename ---
            # Filename format is assumed to be: {img_id}_{ann_id}_cat{cat_id}__{source}.jpg
            # We extract the part after '_cat'
            cat_id_str = glyph_file.stem.split("_cat")[1].split("__")[0]
            category_id = int(cat_id_str)

            manifest_rows.append({
                "path": path,
                "tm_id": tm_id,
                "source": source,
                "category_id": category_id
            })
        except (IndexError, ValueError):
            # This will skip any files that don't match the expected naming convention
            # print(f"Warning: Could not parse category_id from filename: {glyph_file.name}")
            continue

# Write header and all rows to the new CSV
print(f"\nFound {len(manifest_rows)} valid glyphs.")
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "tm_id", "source", "category_id"])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"✅ Full manifest with category IDs saved to: {output_csv.resolve()}")