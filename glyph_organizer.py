import os
import shutil
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

#  Paths 
cv_dir = Path("glyph_crops/train")
kornia_dir = Path("glyph_crops/batch_cropper")
json_path = Path("papytwin/HomerCompTraining/HomerCompTraining.json")
excel_path = Path("papytwin/1.CompetitionOverview.xlsx")
output_dir = Path("glyph_crops_merged_by_tm")
output_dir.mkdir(parents=True, exist_ok=True)

#Load image_id to file_stem from JSON
with open(json_path) as f:
    data = json.load(f)
image_id_to_file = {
    img["id"]: Path(img["file_name"]).stem.lower().strip()
    for img in data["images"]
}

#Load file_stem to TM ID from Excel
df = pd.read_excel(excel_path)
df["Image Name"] = df["Image Name"].astype(str).str.strip().str.lower()
file_to_tm = dict(zip(df["Image Name"], df["TM with READ item name in ()"]))

#Helper to tag and copy
def process_set(glyph_dir, tag, skipped, copy_counter):
    glyph_files = list(glyph_dir.glob("*.jpg"))
    for glyph_path in tqdm(glyph_files, desc=f"Processing {tag}"):
        try:
            glyph_name = glyph_path.stem  # e.g., 6010_32_cat15
            image_id = int(glyph_name.split("_")[0])

            file_stem = image_id_to_file.get(image_id)
            if not file_stem:
                skipped.append((tag, glyph_path.name, "Missing in JSON"))
                continue

            tm_id = file_to_tm.get(file_stem)
            if not tm_id:
                skipped.append((tag, glyph_path.name, f"Missing in Excel (filename: {file_stem})"))
                continue

            tm_folder = output_dir / f"TM_{tm_id}"
            tm_folder.mkdir(exist_ok=True)

            # Add suffix to filename to indicate source
            new_name = f"{glyph_path.stem}__{tag}.jpg"
            shutil.copy(glyph_path, tm_folder / new_name)
            copy_counter[0] += 1

        except Exception as e:
            skipped.append((tag, glyph_path.name, str(e)))

#Process both sets
skipped = []
copy_counter = [0]

process_set(cv_dir, "opencv", skipped, copy_counter)
process_set(kornia_dir, "kornia", skipped, copy_counter)

#Debug samples
print("\nSample entries:")
print("From JSON:", list(image_id_to_file.items())[:5])
print("From Excel:", list(file_to_tm.items())[:5])

#Summary 
print(f"\nTotal glyphs copied: {copy_counter[0]}")
print(f"Total skipped: {len(skipped)}")

if skipped:
    print("\nSkipped samples:")
    for tag, fname, reason in skipped[:10]:  # Show a few
        print(f"  [{tag}] {fname}: {reason}")
