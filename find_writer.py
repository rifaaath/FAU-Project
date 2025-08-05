import json
import pandas as pd
from pathlib import Path

# --- CONFIG ---
# --- ❗️ PASTE THE GLYPH BASENAME YOU ARE INTERESTED IN HERE ---
TARGET_GLYPH_BASENAME = "2_224_cat119"
# --- END OF USER CONFIG ---

# --- Path Config ---
training_json_path = Path("papytwin/HomerCompTraining/HomerCompTraining.json")
excel_path = Path("papytwin/1.CompetitionOverview.xlsx")

# --- Main Script ---

# --- Step 1: Extract the image_id from the basename ---
try:
    target_image_id = int(TARGET_GLYPH_BASENAME.split("_")[0])
    print(f"Extracted Image ID: {target_image_id}")
except (ValueError, IndexError):
    print(f"❌ Error: The basename '{TARGET_GLYPH_BASENAME}' does not seem to follow the expected 'imageid_...' format.")
    exit()

# --- Step 2: Find the document filename from the image_id ---
print(f"Loading metadata from {training_json_path}...")
with open(training_json_path) as f:
    data = json.load(f)

document_filename = None
for image_info in data['images']:
    if image_info['id'] == target_image_id:
        # Get just the filename, not the whole path
        document_filename = Path(image_info['file_name']).name
        print(f"Found corresponding document: '{document_filename}'")
        break

if not document_filename:
    print(f"❌ Error: Could not find Image ID {target_image_id} in the JSON file.")
    exit()

# --- Step 3: Find the TM ID from the document filename ---
print(f"Loading writer metadata from {excel_path}...")
df_meta = pd.read_excel(excel_path)
# Normalize both the lookup key and the column for a robust match
df_meta["Image Name Normalized"] = df_meta["Image Name"].astype(str).str.strip().str.lower()
document_filename_normalized = Path(document_filename).stem.lower().strip()

writer_tm_id = None
# Find the row where the normalized "Image Name" matches our document's stem
matching_row = df_meta[df_meta["Image Name Normalized"] == document_filename_normalized]

if not matching_row.empty:
    # Get the TM ID from that row
    tm_id_raw = matching_row.iloc[0]["TM with READ item name in ()"]
    writer_tm_id = f"TM_{tm_id_raw}"
else:
    print(f"❌ Error: Could not find document '{document_filename}' in the Excel file.")
    exit()


# --- Final Result ---
print("\n" + "="*50)
print("  RESULTS")
print("="*50)
print(f"The glyph '{TARGET_GLYPH_BASENAME}' belongs to:")
print(f"\n    Writer ID: {writer_tm_id}\n")
print("="*50)