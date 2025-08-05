from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
# Paths to the ORIGINAL crop directories, BEFORE they were merged.
opencv_dir = Path("glyph_crops/train")
kornia_dir = Path("glyph_crops/batch_cropper")
# --- END CONFIG ---

print("Scanning for common glyphs in both pipelines...")

# Get the set of all basenames (e.g., '3082_30543_cat119') from each directory
print(f"Scanning OpenCV directory at '{opencv_dir}'...")
opencv_stems = {p.stem for p in tqdm(opencv_dir.glob("*.jpg"))}
print(f"Found {len(opencv_stems)} unique OpenCV glyphs.")

print(f"Scanning Kornia directory at '{kornia_dir}'...")
kornia_stems = {p.stem for p in tqdm(kornia_dir.glob("*.jpg"))}
print(f"Found {len(kornia_stems)} unique Kornia glyphs.")

# --- Find the intersection ---
common_stems = opencv_stems.intersection(kornia_stems)

if not common_stems:
    print("\n❌ Critical Error: No common glyphs found between the two directories.")
    print("This suggests a fundamental issue in how the files were named or processed.")
else:
    print(f"\n✅ Found {len(common_stems)} glyphs that exist in BOTH pipelines.")
    print("You can use any of the following basenames in your 'visualize_specific_pair.py' script:")

    # Print the first 10 examples
    for i, stem in enumerate(list(common_stems)[:10]):
        print(f"  - {stem}")