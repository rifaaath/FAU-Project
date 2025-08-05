import json
from collections import defaultdict
from tqdm import tqdm

# --- CONFIG ---
# ❗️ Correct IDs based on your JSON snippet.
# We now define lists to accept either upper or lower case.
KAPPA_IDS = [49, 33]  # [κ, Κ]
ALPHA_IDS = [165, 8]  # [α, Α]
IOTA_IDS = [105, 212]  # [ι, Ι]

# Input and output paths
predictions_path = "papytwin/predictions/CV_2Twin_res50_box96_train/predictions.json"
output_path = "kai_boxes.json"
# --- END CONFIG ---

print(f"Loading full predictions from: {predictions_path}")
with open(predictions_path) as f:
    data = json.load(f)["annotations"]

print(f"Loaded {len(data)} total predictions.")
print(f"Searching for 'kai' trigrams using IDs:")
print(f"  Kappa (κ/Κ): {KAPPA_IDS}")
print(f"  Alpha (α/Α): {ALPHA_IDS}")
print(f"  Iota  (ι/Ι): {IOTA_IDS}")

# --- Step 1: Group all predictions by their image_id ---
predictions_by_image = defaultdict(list)
for pred in tqdm(data, desc="Grouping predictions by page"):
    predictions_by_image[pred['image_id']].append(pred)

# --- Step 2: Search for the trigram in each page ---
kai_trigram_boxes = []
found_count = 0

for image_id, glyphs in tqdm(predictions_by_image.items(), desc="Searching for trigrams"):
    if len(glyphs) < 3:
        continue

    # Sort glyphs by reading order (left-to-right)
    glyphs.sort(key=lambda g: g['bbox'][0])

    # Slide a window of size 3 across the sorted glyphs
    for i in range(len(glyphs) - 2):
        window = glyphs[i: i + 3]

        # --- ✅ CORRECTED LOGIC: Check if the IDs are in our allowed lists ---
        is_kappa = window[0]['category_id'] in KAPPA_IDS
        is_alpha = window[1]['category_id'] in ALPHA_IDS
        is_iota = window[2]['category_id'] in IOTA_IDS

        # --- Step 2c: Check if the sequence matches ---
        if is_kappa and is_alpha and is_iota:
            # SUCCESS! We found a 'kai' trigram.
            found_count += 1
            # Add the three glyphs that form this trigram to our results
            kai_trigram_boxes.extend(window)

# --- Step 3: Save the results ---
output_data = {"annotations": kai_trigram_boxes}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"\n✅ Success! Found {found_count} 'kai' trigrams.")
print(f"✅ Saved {len(kai_trigram_boxes)} corresponding bounding boxes to '{output_path}'")