import csv
from pathlib import Path

# Set path to merged dataset
merged_dir = Path("glyph_crops_merged_by_tm")
output_csv = Path("glyph_manifest.csv")
print("Hello")
# merged_dir = Path("glyph_crops/alpha")
# output_csv = Path("alpha_manifest.csv")

# Write header + entries
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["path", "tm_id", "source"])

    for tm_folder in merged_dir.iterdir():
        if not tm_folder.is_dir():
            continue

        tm_id = tm_folder.name  # e.g., TM_123456
        for glyph_file in tm_folder.glob("*.jpg"):
            path = glyph_file.as_posix()
            source = "kornia" if "__kornia" in glyph_file.name else "opencv"
            writer.writerow([path, tm_id, source])

print(f"Manifest saved to: {output_csv.resolve()}")
