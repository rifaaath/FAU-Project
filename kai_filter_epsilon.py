import json
from collections import defaultdict
from tqdm import tqdm

#  CONFIG 

epsilon_IDS = [103, 23]


# Input and output paths
predictions_path = "papytwin/predictions/CV_2Twin_res50_box96_train/predictions.json"
output_path = "epsilon_boxes.json"
#  END CONFIG 

print(f"Loading full predictions from: {predictions_path}")
with open(predictions_path) as f:
    data = json.load(f)["annotations"]

print(f"Loaded {len(data)} total predictions.")
print(f"Searching for 'epsilon' using IDs:")
print(f"  Kappa (κ/Κ): {epsilon_IDS}")


#  Step 1: Group all predictions by their image_id 
predictions_by_image = defaultdict(list)
for pred in tqdm(data, desc="Grouping predictions by page"):
    predictions_by_image[pred['image_id']].append(pred)

#  Step 2: Search for the trigram in each page 
epsilon_boxes = []
found_count = 0

for image_id, glyphs in tqdm(predictions_by_image.items(), desc="Searching for epsilons"):
    if len(glyphs) < 3:
        continue

    # Sort glyphs by reading order (left-to-right)
    glyphs.sort(key=lambda g: g['bbox'][0])

    # Slide a window of size 3 across the sorted glyphs
    for i in range(len(glyphs) - 2):
        window = glyphs[i: i + 3]

        #  ✅ CORRECTED LOGIC: Check if the IDs are in our allowed lists 
        is_epsilon = window[0]['category_id'] in epsilon_IDS

        #  Step 2c: Check if the sequence matches 
        if is_epsilon:
            # SUCCESS! We found a 'epsilon' trigram.
            found_count += 1
            # Add the three glyphs that form this trigram to our results
            epsilon_boxes.extend(window)

#  Step 3: Save the results 
output_data = {"annotations": epsilon_boxes}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"\n✅ Success! Found {found_count} epsilons.")
print(f"✅ Saved {len(epsilon_boxes)} corresponding bounding boxes to '{output_path}'")