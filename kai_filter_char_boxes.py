import json
from tqdm import tqdm

# --- CONFIG ---
# ID of the character you want to isolate (e.g., 1 for alpha 'α')
TARGET_CATEGORY_ID = 23
TARGET_CHAR_NAME = "epsilon" # For naming the output file

# Input and output paths
predictions_path = "papytwin/predictions/CV_2Twin_res50_box96_train/predictions.json"
output_path = f"{TARGET_CHAR_NAME}_boxes.json"
# --- END CONFIG ---

print(f"Loading full predictions from: {predictions_path}")
with open(predictions_path) as f:
    # We only need the 'annotations' part
    data = json.load(f)["annotations"]

print(f"Loaded {len(data)} total predictions.")
print(f"Filtering for character ID: {TARGET_CATEGORY_ID} ({TARGET_CHAR_NAME})...")

filtered_boxes = []
for pred in tqdm(data, desc="Filtering boxes"):
    if pred["category_id"] == TARGET_CATEGORY_ID:
        filtered_boxes.append(pred)

# The output format should match the original for compatibility
output_data = {"annotations": filtered_boxes}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"\n✅ Success! Saved {len(filtered_boxes)} bounding boxes for '{TARGET_CHAR_NAME}' to '{output_path}'")